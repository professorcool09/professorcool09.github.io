import os, math, pickle, json
import numpy as np
from flask import Flask, jsonify, request, render_template

from catanatron import Game, Color, ActionType
from catanatron.models.player import RandomPlayer, Player
from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, LandTile

from catanatron_gym.features import create_sample, get_feature_ordering
from catanatron_gym.envs.catanatron_env import (
    to_action_space, from_action_space, normalize_action, ACTIONS_ARRAY
)
from sb3_contrib.ppo_mask import MaskablePPO

# ── Monkey-patch roll_dice to capture dice values ─────────────────────────────
import catanatron.state as _catan_state
_original_roll_dice = _catan_state.roll_dice
_last_dice = [None]   # list so the closure can mutate it

def _patched_roll_dice():
    result = _original_roll_dice()
    _last_dice[0] = result
    return result

_catan_state.roll_dice = _patched_roll_dice

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "MYMODEL_final5000000")
VEC_PATH   = os.path.join(BASE_DIR, "vec_normalize.pkl")
HEX_SIZE   = 72          # must match TC constant in frontend JS
FEATURES   = get_feature_ordering(num_players=2)

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading PPO model …")
model = MaskablePPO.load(MODEL_PATH)
print("Loading VecNormalize stats …")
with open(VEC_PATH, "rb") as f:
    vec_norm = pickle.load(f)
print("Ready.")

# ── Board geometry helpers ────────────────────────────────────────────────────
def cube_to_pixel(q, r, size=HEX_SIZE):
    x = size * (3 / 2 * q)
    y = size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return x, y

ANGLE_MAP = {
    "NORTHEAST":   0,
    "NORTH":      60,
    "NORTHWEST":  120,
    "SOUTHWEST":  180,
    "SOUTH":      240,
    "SOUTHEAST":  300,
}

def compute_node_positions(board):
    positions = {}
    for (q, r, _s), tile in board.map.land_tiles.items():
        cx, cy = cube_to_pixel(q, r)
        for ref, node_id in tile.nodes.items():
            if node_id not in positions:
                angle_rad = math.radians(ANGLE_MAP[ref.value])
                positions[node_id] = (
                    cx + HEX_SIZE * math.cos(angle_rad),
                    cy + HEX_SIZE * math.sin(angle_rad),
                )
    return positions

# ── Game session ──────────────────────────────────────────────────────────────
game_session = {}

def init_game():
    human_color  = Color.BLUE
    bot_color    = Color.RED
    human_player = Player(human_color)
    bot_player   = RandomPlayer(bot_color)
    g = Game([human_player, bot_player])
    node_pos = compute_node_positions(g.state.board)
    _last_dice[0] = None   # clear dice on new game
    game_session.update(
        game=g,
        human_color=human_color,
        bot_color=bot_color,
        node_pos=node_pos,
        log=[],
        last_dice=None,
    )
    return g

def normalize_obs(obs_raw):
    obs  = np.array(obs_raw, dtype=np.float64)
    mean = vec_norm.obs_rms.mean
    var  = vec_norm.obs_rms.var
    eps  = vec_norm.epsilon if hasattr(vec_norm, "epsilon") else 1e-8
    clip = vec_norm.clip_obs
    obs  = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip)
    return obs.astype(np.float32)

def bot_decide(game, bot_color):
    sample  = create_sample(game, bot_color)
    obs_raw = np.array([float(sample[f]) for f in FEATURES], dtype=np.float32)
    obs     = normalize_obs(obs_raw)

    n    = model.action_space.n
    mask = np.zeros(n, dtype=bool)
    idx_to_action = {}
    for a in game.state.playable_actions:
        try:
            idx = to_action_space(a)
            if 0 <= idx < n:
                mask[idx] = True
                idx_to_action[idx] = a
        except Exception:
            pass

    if not idx_to_action:
        return game.state.playable_actions[0]

    action_idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return idx_to_action.get(int(action_idx), list(idx_to_action.values())[0])

# ── Serialisation helpers ─────────────────────────────────────────────────────
def resource_str(r):
    if r is None:
        return "DESERT"
    return str(r).replace("Resource.", "")

def color_str(c):
    if c is None:
        return None
    return str(c).replace("Color.", "").replace("<Color.", "").replace(">", "").upper()

RES_EMOJI = {
    "WOOD":  "\U0001fab5",
    "BRICK": "\U0001f9f1",
    "SHEEP": "\U0001f411",
    "WHEAT": "\U0001f33e",
    "ORE":   "\u26cf\ufe0f",
}

def trade_label(action):
    """Format a MARITIME_TRADE action value as e.g. '4 🪵 → 1 🧱'"""
    v = action.value
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return f"Trade {v}"
    give_res = v[0]  # first element is always the resource being given
    get_res  = v[-1]  # last element is always the resource being received
    give_qty = sum(1 for x in v[:-1] if x is not None)
    give_emoji = RES_EMOJI.get(str(give_res), str(give_res))
    get_emoji  = RES_EMOJI.get(str(get_res),  str(get_res))
    return f"{give_qty} {give_emoji}  \u2192  1 {get_emoji}"

def action_label(action):
    at = action.action_type
    v  = action.value
    if at == ActionType.ROLL:              return "Roll Dice"
    if at == ActionType.END_TURN:          return "End Turn"
    if at == ActionType.BUILD_SETTLEMENT:  return f"Build Settlement at node {v}"
    if at == ActionType.BUILD_CITY:        return f"Build City at node {v}"
    if at == ActionType.BUILD_ROAD:        return f"Build Road on edge {v}"
    if at == ActionType.BUY_DEVELOPMENT_CARD: return "Buy Dev Card"
    if at == ActionType.MOVE_ROBBER:       return f"Move Robber to {v}"
    if at == ActionType.DISCARD:           return "Discard"
    if at == ActionType.PLAY_KNIGHT_CARD:  return "Play Knight"
    if at == ActionType.PLAY_YEAR_OF_PLENTY: return f"Year of Plenty: {v}"
    if at == ActionType.PLAY_MONOPOLY:     return f"Monopoly: {v}"
    if at == ActionType.PLAY_ROAD_BUILDING: return "Road Building"
    if at == ActionType.MARITIME_TRADE:    return trade_label(action)
    return str(at)

def serialize_state():
    g        = game_session["game"]
    board    = g.state.board
    ps       = g.state.player_state
    human    = game_session["human_color"]
    bot      = game_session["bot_color"]
    node_pos = game_session["node_pos"]

    # FIX: use color_to_index so P0/P1 always matches the right color
    # regardless of who went first this game
    human_prefix = f"P{g.state.color_to_index[human]}"
    bot_prefix   = f"P{g.state.color_to_index[bot]}"

    # Tiles
    tiles = []
    for (q, r, s), tile in board.map.land_tiles.items():
        cx, cy = cube_to_pixel(q, r)
        tiles.append({
            "id":         tile.id,
            "q": q, "r": r, "s": s,
            "cx": cx, "cy": cy,
            "resource":   resource_str(tile.resource),
            "number":     tile.number,
            "has_robber": board.robber_coordinate == (q, r, s),
        })

    # Nodes
    nodes = []
    for nid, (nx, ny) in node_pos.items():
        building = board.buildings.get(nid)
        btype = bcolor = None
        if building is not None:
            bcolor = color_str(building[0])
            btype  = str(building[1]).replace("BuildingType.", "")
        nodes.append({"id": nid, "x": nx, "y": ny,
                      "building_type": btype, "color": bcolor})

    # Edges
    edges = []
    for (n1, n2) in get_edges():
        key    = tuple(sorted((n1, n2)))
        rc     = board.roads.get(key) or board.roads.get((n1, n2)) or board.roads.get((n2, n1))
        rcolor = color_str(rc) if rc is not None else None
        x1, y1 = node_pos[n1]
        x2, y2 = node_pos[n2]
        edges.append({"n1": n1, "n2": n2,
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "color": rcolor})

    # Ports
    ports = []
    for pid, port in board.map.ports_by_id.items():
        land_nodes = [n for n in port.nodes.values() if n in node_pos]
        if len(land_nodes) < 2:
            continue
        best, best_dist = (land_nodes[0], land_nodes[1]), float('inf')
        for i in range(len(land_nodes)):
            for j in range(i + 1, len(land_nodes)):
                n1, n2 = land_nodes[i], land_nodes[j]
                x1, y1 = node_pos[n1]; x2, y2 = node_pos[n2]
                d = (x2-x1)**2 + (y2-y1)**2
                if d < best_dist:
                    best_dist = d; best = (n1, n2)
        ports.append({"resource": resource_str(port.resource), "nodes": list(best)})

    # Player states — now using correct prefix per color
    def pstate(prefix):
        return {
            "vp":       ps.get(f"{prefix}_ACTUAL_VICTORY_POINTS", 0),
            "wood":     ps.get(f"{prefix}_WOOD_IN_HAND", 0),
            "brick":    ps.get(f"{prefix}_BRICK_IN_HAND", 0),
            "sheep":    ps.get(f"{prefix}_SHEEP_IN_HAND", 0),
            "wheat":    ps.get(f"{prefix}_WHEAT_IN_HAND", 0),
            "ore":      ps.get(f"{prefix}_ORE_IN_HAND", 0),
            "has_road": bool(ps.get(f"{prefix}_HAS_ROAD", False)),
            "has_army": bool(ps.get(f"{prefix}_HAS_ARMY", False)),
            # Individual dev card counts
            "knight_cards":   ps.get(f"{prefix}_KNIGHT_IN_HAND", 0),
            "vp_cards":       ps.get(f"{prefix}_VICTORY_POINT_IN_HAND", 0),
            "year_cards":     ps.get(f"{prefix}_YEAR_OF_PLENTY_IN_HAND", 0),
            "monopoly_cards": ps.get(f"{prefix}_MONOPOLY_IN_HAND", 0),
            "road_cards":     ps.get(f"{prefix}_ROAD_BUILDING_IN_HAND", 0),
            "dev_cards": (
                ps.get(f"{prefix}_KNIGHT_IN_HAND", 0) +
                ps.get(f"{prefix}_YEAR_OF_PLENTY_IN_HAND", 0) +
                ps.get(f"{prefix}_MONOPOLY_IN_HAND", 0) +
                ps.get(f"{prefix}_ROAD_BUILDING_IN_HAND", 0) +
                ps.get(f"{prefix}_VICTORY_POINT_IN_HAND", 0)
            ),
        }

    # Dice — use captured value from monkey-patch
    last_roll = _last_dice[0]
    dice_info = {"d1": last_roll[0], "d2": last_roll[1], "total": sum(last_roll)} if last_roll else None

    # Valid actions
    valid_actions = []
    for a in g.state.playable_actions:
        try:
            idx = to_action_space(a)
        except Exception:
            idx = -1
        valid_actions.append({
            "idx":   idx,
            "type":  str(a.action_type).replace("ActionType.", ""),
            "value": str(a.value),
            "label": action_label(a),
            "color": color_str(a.color),
        })

    winner = g.winning_color()

    return {
        "tiles":         tiles,
        "nodes":         nodes,
        "edges":         edges,
        "ports":         ports,
        "human_color":   color_str(human),
        "bot_color":     color_str(bot),
        "human":         pstate(human_prefix),
        "bot":           pstate(bot_prefix),
        "current_color": color_str(g.state.current_color()),
        "is_human_turn": g.state.current_color() == human,
        "valid_actions": valid_actions,
        "num_turns":     g.state.num_turns,
        "winner":        color_str(winner) if winner else None,
        "log":           game_session["log"][-20:],
        "is_initial_phase": g.state.is_initial_build_phase,
        "dice":          dice_info,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", hex_size=HEX_SIZE)

@app.route("/api/new_game", methods=["POST"])
def new_game():
    init_game()
    game_session["log"] = ["New game started! You are BLUE. Bot is RED."]
    _run_bot_turns()
    return jsonify(serialize_state())

@app.route("/api/state", methods=["GET"])
def get_state():
    if "game" not in game_session:
        init_game()
        game_session["log"] = ["New game started! You are BLUE. Bot is RED."]
        _run_bot_turns()
    return jsonify(serialize_state())

@app.route("/api/action", methods=["POST"])
def human_action():
    data       = request.get_json()
    action_idx = int(data.get("action_idx", -1))
    g          = game_session["game"]
    human      = game_session["human_color"]

    if g.winning_color() is not None:
        return jsonify({"error": "Game is over"}), 400
    if g.state.current_color() != human:
        return jsonify({"error": "Not your turn"}), 400

    chosen = None
    for a in g.state.playable_actions:
        try:
            if to_action_space(a) == action_idx:
                chosen = a; break
        except Exception:
            pass

    if chosen is None:
        return jsonify({"error": "Invalid action"}), 400

    game_session["log"].append(f"You: {action_label(chosen)}")
    g.execute(chosen)
    _run_bot_turns()
    return jsonify(serialize_state())

def _run_bot_turns():
    g   = game_session["game"]
    bot = game_session["bot_color"]
    for _ in range(200):
        if g.winning_color() is not None or g.state.current_color() != bot:
            break
        action = bot_decide(g, bot)
        game_session["log"].append(f"Bot: {action_label(action)}")
        g.execute(action)

@app.route("/api/bot_move", methods=["POST"])
def force_bot_move():
    g   = game_session["game"]
    bot = game_session["bot_color"]
    if g.winning_color() is not None:
        return jsonify({"error": "Game over"}), 400
    if g.state.current_color() != bot:
        return jsonify({"error": "Not bot's turn"}), 400
    action = bot_decide(g, bot)
    game_session["log"].append(f"Bot: {action_label(action)}")
    g.execute(action)
    return jsonify(serialize_state())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
