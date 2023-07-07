"""gridworld interface"""

import ctypes
import importlib
import os
from typing import List, Tuple

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class EventNode:
    """an AST node of the event expression"""

    OP_AND = 0
    OP_OR = 1
    OP_NOT = 2

    OP_KILL = 3
    OP_AT = 4
    OP_IN = 5
    OP_COLLIDE = 6
    OP_ATTACK = 7
    OP_DIE = 8
    OP_IN_A_LINE = 9
    OP_ALIGN = 10

    # can extend more operation below

    def __init__(self):
        # for non-leaf node
        self.op = None
        # for leaf node
        self.predicate = None

        self.inputs = []

    def __call__(self, subject, predicate, *args):
        node = EventNode()
        node.predicate = predicate
        if predicate == "kill":
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == "at":
            node.op = EventNode.OP_AT
            coor = args[0]
            node.inputs = [subject, coor[0], coor[1]]
        elif predicate == "in":
            node.op = EventNode.OP_IN
            coor = args[0]
            x1, y1 = min(coor[0][0], coor[1][0]), min(coor[0][1], coor[1][1])
            x2, y2 = max(coor[0][0], coor[1][0]), max(coor[0][1], coor[1][1])
            node.inputs = [subject, x1, y1, x2, y2]
        elif predicate == "attack":
            node.op = EventNode.OP_ATTACK
            node.inputs = [subject, args[0]]
        elif predicate == "collide":
            node.op = EventNode.OP_COLLIDE
            node.inputs = [subject, args[0]]
        elif predicate == "die":
            node.op = EventNode.OP_DIE
            node.inputs = [subject]
        elif predicate == "in_a_line":
            node.op = EventNode.OP_IN_A_LINE
            node.inputs = [subject]
        elif predicate == "align":
            node.op = EventNode.OP_ALIGN
            node.inputs = [subject]
        else:
            raise Exception("invalid predicate of event " + predicate)
        return node

    def __and__(self, other):
        node = EventNode()
        node.op = EventNode.OP_AND
        node.inputs = [self, other]
        return node

    def __or__(self, other):
        node = EventNode()
        node.op = EventNode.OP_OR
        node.inputs = [self, other]
        return node

    def __invert__(self):
        node = EventNode()
        node.op = EventNode.OP_NOT
        node.inputs = [self]
        return node


Event = EventNode()


class AgentSymbol:
    """Symbol to represent some agents in defining events."""

    def __init__(self, group: ctypes.c_int32, index):
        """Define an agent symbol. It can be the object or subject of EventNode.

        Args:
            group (ctypes.c_int32): group handle
            index (int or str): int: a deterministic integer id.
                str: can be 'all' or 'any', represents all or any agents in a group.
        """
        self.group = group if group is not None else -1
        if index == "any":
            self.index = -1
        elif index == "all":
            self.index = -2
        else:
            assert isinstance(self.index, int), "index must be a deterministic int"
            self.index = index

    def __str__(self):
        return "agent(%d,%d)" % (self.group, self.index)


class Config:
    """Configuration class for a Gridworld game."""

    def __init__(self):
        self.config_dict = {}
        self.agent_type_dict = {}
        self.groups = []
        self.reward_rules = []

    def set(self, args: dict):
        """Set parameters of global configuration.

        :param dict args:

        Contains the following configuration attributes:
            * **map_width** (*int*): Number of horizontal grid squares in the Gridworld.
            * **map_height** (*int*): Number of vertical grid squares in the Gridworld.
            * **embedding_size** (*int*): Embedding size for the observation features.
            * **render_dir** (*str*): Directory to save render file.
            * **seed** (*int*): Random seed.
            * **food_mode** (*bool*): Dead agents drop food on the map.
            * **turn_mode** (*bool*): Include 2 more actions -- turn left and turn right.
            * **minimap_mode** (*bool*): Include minimap in observations.
        """
        for key in args:
            self.config_dict[key] = args[key]

    def register_agent_type(self, name: str, attr: dict):
        """Register an agent type.

        :param str name: Name of the type (should be unique).
        :param dict attr:

        Contains the following configuration attributes:
            * **height** (*int*):   Height of agent body.
            * **width** (*int*):    Width of agent body.
            * **speed** (*float*):  Maximum speed, i.e. the radius of move circle of the agent.
            * **hp** (*float*):    Maximum health point of the agent.
            * **view_range** (*gw.CircleRange* or *gw.SectorRange*): Field of view of the agent.
            * **damage** (*float*):         Attack damage.
            * **step_recover** (*float*):   Step recover (healing) of health points (can be negative).
            * **kill_supply** (*float*):    Hp gain for killing this type of agent.
            * **step_reward** (*float*):    Reward gained in every step.
            * **kill_reward** (*float*):    Reward gained for killing this type of agent.
            * **dead_penalty** (*float*):   Reward gained for dying.
            * **attack_penalty** (*float*): Reward gained when performing an attack (to discourage attacking empty grid cells).

        :Returns:
            name (str): Name of the type.
        """
        if name in self.agent_type_dict:
            raise Exception("type name %s already exists" % name)
        self.agent_type_dict[name] = attr
        return name

    def add_group(self, agent_type: str):
        """Add a group to the configuration.

        Args:
            agent_type (str): Name of agent type contained in this group.
        Returns:
            group_handle (int): Handle for the new group.
        """
        no = len(self.groups)
        self.groups.append(agent_type)
        return no

    def add_reward_rule(
        self,
        on: EventNode,
        receiver: List[AgentSymbol],
        value: List[float],
        terminal: bool = False,
    ):
        """Add a reward rule.

        Args:
            on (Event): An objecting representing a bool expression of the trigger event.
            receiver (List[AgentSymbol]): Receiver of this reward rule. If the receiver is not a deterministic agent,
                it must be one of the agents involved in the triggering event.
            value List[float]: Value to assign.
            terminal (bool): Whether this event will terminate the game.

        """
        if not (isinstance(receiver, tuple) or isinstance(receiver, list)):
            assert not (isinstance(value, tuple) or isinstance(value, tuple))
            receiver = [receiver]
            value = [value]
        if len(receiver) != len(value):
            raise Exception("the length of receiver and value should be equal")
        self.reward_rules.append([on, receiver, value, terminal])


class GridWorld(Environment):
    """
    The main MAgent2 class for implementing environments. MAgent2 environments are square Gridworlds wherein each coordinate may contain an agent, a wall, or nothing.

    The class attributes are not accessible directly due to them living in the underlying C++ code.
    Thus, there are get/set methods for retrieving and manipulating their values.
    """

    # constant
    OBS_INDEX_VIEW = 0
    OBS_INDEX_HP = 1

    def __init__(self, config: Config, **kwargs):
        """
        Parameters
        ----------
        config: str or Config Object
            if config is a string, then it is a name of builtin config,
                builtin config are stored in python/magent/builtin/config
                kwargs are the arguments to the config
            if config is a Config Object, then parameters are stored in that object
        """
        Environment.__init__(self)

        # if is str, load built in configuration
        if isinstance(config, str):
            # built-in config are stored in python/magent/builtin/config
            try:
                demo_game = importlib.import_module("magent2.builtin.config." + config)
                config = getattr(demo_game, "get_config")(**kwargs)
            except AttributeError:
                raise BaseException('unknown built-in game "' + config + '"')

        # create new game
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"GridWorld")
        self.game = game

        # set global configuration
        config_value_type = {
            "map_width": int,
            "map_height": int,
            "food_mode": bool,
            "turn_mode": bool,
            "minimap_mode": bool,
            "revive_mode": bool,
            "goal_mode": bool,
            "embedding_size": int,
            "render_dir": str,
            "seed": int,
        }

        for key in config.config_dict:
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(
                    self.game,
                    key.encode("ascii"),
                    ctypes.byref(ctypes.c_int(config.config_dict[key])),
                )
            elif value_type is bool:
                _LIB.env_config_game(
                    self.game,
                    key.encode("ascii"),
                    ctypes.byref(ctypes.c_bool(config.config_dict[key])),
                )
            elif value_type is float:
                _LIB.env_config_game(
                    self.game,
                    key.encode("ascii"),
                    ctypes.byref(ctypes.c_float(config.config_dict[key])),
                )
            elif value_type is str:
                _LIB.env_config_game(
                    self.game,
                    key.encode("ascii"),
                    ctypes.c_char_p(config.config_dict[key]),
                )

        # register agent types
        for name in config.agent_type_dict:
            type_args = config.agent_type_dict[name]

            # special pre-process for view range and attack range
            for key in [x for x in type_args.keys()]:
                if key == "view_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["view_radius"] = val.radius
                    type_args["view_angle"] = val.angle
                elif key == "attack_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["attack_radius"] = val.radius
                    type_args["attack_angle"] = val.angle

            length = len(type_args)
            keys = (ctypes.c_char_p * length)(
                *[key.encode("ascii") for key in type_args.keys()]
            )
            values = (ctypes.c_float * length)(*type_args.values())

            _LIB.gridworld_register_agent_type(
                self.game, name.encode("ascii"), length, keys, values
            )

        # serialize event expression, send to C++ engine
        self._serialize_event_exp(config)

        # init group handles
        self.group_handles = []
        for item in config.groups:
            handle = ctypes.c_int32()
            _LIB.gridworld_new_group(
                self.game, item.encode("ascii"), ctypes.byref(handle)
            )
            self.group_handles.append(handle)

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view space, feature space, action space
        self.view_space = {}
        self.feature_space = {}
        self.action_space = {}
        buf = np.empty((3,), dtype=np.int32)
        for handle in self.group_handles:
            _LIB.env_get_info(
                self.game,
                handle,
                b"view_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.view_space[handle.value] = (buf[0], buf[1], buf[2])
            _LIB.env_get_info(
                self.game,
                handle,
                b"feature_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.feature_space[handle.value] = (buf[0],)
            _LIB.env_get_info(
                self.game,
                handle,
                b"action_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.action_space[handle.value] = (buf[0],)

    def reset(self):
        """Resets the environment to an initial internal state."""
        _LIB.env_reset(self.game)

    def add_walls(self, method: str, **kwargs):
        """Adds walls to the environment.

        Args:
            method (str): Can be 'random' or 'custom'. If method is 'random', then kwargs["n"] is an int.
                If method is 'custom', then kwargs["pos"] is a list of coordination

        ```
        # add 1000 walls randomly
        >>> env.add_walls(method="random", n=1000)

        # add 3 walls to (1,2), (4,5) and (9, 8) in map
        >>> env.add_walls(method="custom", pos=[(1,2), (4,5), (9,8)])
        ```
        """
        # handle = -1 for walls
        kwargs["dir"] = 0
        self.add_agents(-1, method, **kwargs)

    # ====== AGENT ======
    def new_group(self, name: str) -> ctypes.c_int32:
        """Registers a new group of agents into environment.

        Args:
            name (str): Name of the group.

        Returns:
            handle (ctypes.c_int32): A handle to reference the group in future gets and sets.

        """
        handle = ctypes.c_int32()
        _LIB.gridworld_new_group(
            self.game, ctypes.c_char_p(name.encode("ascii")), ctypes.byref(handle)
        )
        return handle

    def add_agents(self, handle: ctypes.c_int32, method: str, **kwargs):
        """Adds agents to environment.

        Args:
            handle (ctypes.c_int32): The handle of the group to which to add the agents.
            method (str): Can be 'random' or 'custom'. If method is 'random', then kwargs["n"] is a int.
                If method is 'custom', then kwargs["pos"] is a list of coordination.

        ```
        # add 1000 walls randomly
        >>> env.add_agents(handle, method="random", n=1000)

        # add 3 agents to (1,2), (4,5) and (9, 8) in map
        >>> env.add_agents(handle, method="custom", pos=[(1,2), (4,5), (9,8)])
        ```
        """
        if method == "random":
            _LIB.gridworld_add_agents(
                self.game, handle, int(kwargs["n"]), b"random", 0, 0, 0
            )
        elif method == "custom":
            n = len(kwargs["pos"])
            pos = np.array(kwargs["pos"], dtype=np.int32)
            if len(pos) <= 0:
                return
            if pos.shape[1] == 3:  # if has dir
                xs, ys, dirs = pos[:, 0], pos[:, 1], pos[:, 2]
            else:  # if do not has dir, use zero padding
                xs, ys, dirs = pos[:, 0], pos[:, 1], np.zeros((n,), dtype=np.int32)
            # copy again, to make these arrays continuous in memory
            xs, ys, dirs = np.array(xs), np.array(ys), np.array(dirs)
            _LIB.gridworld_add_agents(
                self.game,
                handle,
                n,
                b"custom",
                as_int32_c_array(xs),
                as_int32_c_array(ys),
                as_int32_c_array(dirs),
            )
        elif method == "fill":
            x, y = kwargs["pos"][0], kwargs["pos"][1]
            width, height = kwargs["size"][0], kwargs["size"][1]
            dir = kwargs.get("dir", np.zeros_like(x))
            bind = np.array([x, y, width, height, dir], dtype=np.int32)
            _LIB.gridworld_add_agents(
                self.game, handle, 0, b"fill", as_int32_c_array(bind), 0, 0, 0
            )
        elif method == "maze":
            # TODO: implement maze add
            x_start, y_start, x_end, y_end = (
                kwargs["pos"][0],
                kwargs["pos"][1],
                kwargs["pos"][2],
                kwargs["pos"][3],
            )
            thick = kwargs["pos"][4]
            bind = np.array([x_start, y_start, x_end, y_end, thick], dtype=np.int32)
            _LIB.gridworld_add_agents(
                self.game, handle, 0, b"maze", as_int32_c_array(bind), 0, 0, 0
            )
        else:
            print("Unknown type of position")
            exit(-1)

    # ====== RUN ======
    def _get_obs_buf(self, group, key, shape, dtype):
        """get buffer to receive observation from c++ engine"""
        obs_buf = self.obs_bufs[key]
        if group in obs_buf:
            ret = obs_buf[group]
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
        else:
            ret = obs_buf[group] = np.empty(shape=shape, dtype=dtype)

        return ret

    def _init_obs_buf(self):
        """init observation buffer"""
        self.obs_bufs = []
        self.obs_bufs.append({})
        self.obs_bufs.append({})

    def get_observation(self, handle: ctypes.c_int32) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the observation for each agent in a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            obs (Tuple[np.ndarray, np.ndarray]): (views, features)
                Views is a numpy array whose shape is n * view_width * view_height * n_channel.
                Features is a numpy array whose shape is n * feature_size.
                For agent i, (views[i], features[i]) is its observation at this step.
        """
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(
            no, self.OBS_INDEX_VIEW, (n,) + view_space, np.float32
        )
        feature_buf = self._get_obs_buf(
            no, self.OBS_INDEX_HP, (n,) + feature_space, np.float32
        )

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)

        return view_buf, feature_buf

    def set_action(self, handle: ctypes.c_int32, actions: np.ndarray):
        """Set actions for whole group.

        Args:
            handle (ctypes.c_int32): Group handle.
            actions (np.ndarray): Array of actions, 1 per agent. The dtype must be int32.
        """
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.int32
        _LIB.env_set_action(
            self.game, handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )

    def step(self):
        """Runs one timestep of the environment using the agents' actions.

        Returns:
            done (bool): Flag indicating whether the game is done or not.
        """
        done = ctypes.c_int32()
        _LIB.env_step(self.game, ctypes.byref(done))
        return bool(done)

    def get_reward(self, handle: ctypes.c_int32) -> np.ndarray:
        """Returns the rewards for all agents in a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            rewards (np.ndarray[float32]): Rewards for all agents in the group.
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.float32)
        _LIB.env_get_reward(
            self.game, handle, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return buf

    def clear_dead(self):
        """Clears dead agents in the engine. Must be called after step()."""
        _LIB.gridworld_clear_dead(self.game)

    # ====== INFO ======
    def get_handles(self) -> List[ctypes.c_int32]:
        """Returns all group handles in the environment.

        Returns:
            handles (List[ctypes.c_int32]): All group handles in the environment.
        """
        return self.group_handles

    def get_num(self, handle: ctypes.c_int32) -> int:
        """Returns the number of agents in a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            num (int): Number of agents in the group.
        """
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b"num", ctypes.byref(num))
        return num.value

    def get_action_space(self, handle: ctypes.c_int32) -> Tuple[int]:
        """Returns the action space for a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            action_space (Tuple[int]): Action space for the group.
        """
        return self.action_space[handle.value]

    def get_view_space(self, handle: ctypes.c_int32) -> Tuple[int, int, int]:
        """Returns the view space for a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            view_space (Tuple[int, int, int]): View space for the group.
        """
        return self.view_space[handle.value]

    def get_feature_space(self, handle: ctypes.c_int32) -> Tuple[int]:
        """Returns the feature space for a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            feature_space (Tuple[int]): Feature space for the group.
        """
        return self.feature_space[handle.value]

    def get_agent_id(self, handle: ctypes.c_int32) -> np.ndarray:
        """Returns the ids of all agents in the group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            ids (np.ndarray[int32]): Ids of all agents in the group.
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.int32)
        _LIB.env_get_info(
            self.game, handle, b"id", buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
        return buf

    def get_alive(self, handle: ctypes.c_int32) -> np.ndarray:
        """Returns the alive status of all agents in a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            alives (np.ndarray[bool]): Whether the agents are alive or not.
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=bool)
        _LIB.env_get_info(
            self.game,
            handle,
            b"alive",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        )
        return buf

    def get_pos(self, handle: ctypes.c_int32) -> np.ndarray:
        """Returns the positions of all agents in a group.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            pos (np.ndarray[int]): The positions of all agents in the group.
                The shape is (n, 2).
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(
            self.game,
            handle,
            b"pos",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return buf

    def get_view2attack(self, handle: ctypes.c_int32) -> Tuple[int, np.ndarray]:
        """Get a matrix with the same size of view_range.
        If element >= 0, then it is an attackable point, and the corresponding
        action number is the value of that element.

        Args:
            handle (ctypes.c_int32): Group handle.

        Returns:
            attack_base (int): Attack action base value.
            buf (np.ndarray): Map attack action into view.
        """
        size = self.get_view_space(handle)[0:2]
        buf = np.empty(size, dtype=np.int32)
        attack_base = ctypes.c_int32()
        _LIB.env_get_info(
            self.game,
            handle,
            b"view2attack",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        _LIB.env_get_info(self.game, handle, b"attack_base", ctypes.byref(attack_base))
        return attack_base.value, buf

    def get_global_minimap(self, height: int, width: int) -> np.ndarray:
        """Compress global map into a minimap of given size.

        Args:
            height (int): Height of minimap.
            width (int): Width of minimap.

        Returns:
            minimap (np.ndarray): Map of shape (n_group + 1, height, width).
        """
        buf = np.empty((height, width, len(self.group_handles)), dtype=np.float32)
        buf[0, 0, 0] = height
        buf[0, 0, 1] = width
        _LIB.env_get_info(
            self.game,
            -1,
            b"global_minimap",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return buf

    def set_seed(self, seed: int):
        """Set random seed of the engine.

        Args:
            seed (int): Seed value.

        """
        _LIB.env_config_game(self.game, b"seed", ctypes.byref(ctypes.c_int(seed)))

    # ====== RENDER ======
    def set_render_dir(self, name: str):
        """Sets the directory to save render file.

        Args:
            name (str): Name of render directory."""
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name.encode("ascii"))

    def render(self):
        """Renders a step."""
        _LIB.env_render(self.game)

    def _get_groups_info(self):
        """private method, for interactive application"""
        n = len(self.group_handles)
        buf = np.empty((n, 5), dtype=np.int32)
        _LIB.env_get_info(
            self.game,
            -1,
            b"groups_info",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return buf

    def _get_walls_info(self):
        """private method, for interactive application"""
        n = 100 * 100
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(
            self.game,
            -1,
            b"walls_info",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        n = buf[0, 0]  # the first line is the number of walls
        return buf[1 : 1 + n]

    def _get_render_info(self, x_range, y_range):
        """private method, for interactive application"""
        n = 0
        for handle in self.group_handles:
            n += self.get_num(handle)

        buf = np.empty((n + 1, 4), dtype=np.int32)
        buf[0] = x_range[0], y_range[0], x_range[1], y_range[1]
        _LIB.env_get_info(
            self.game,
            -1,
            b"render_window_info",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )

        # the first line is for the number of agents in the window range
        info_line = buf[0]
        # agent_ct, attack_event_ct = info_line[0], info_line[1]
        attack_event_ct = info_line[1]
        buf = buf[1 : 1 + info_line[0]]

        agent_info = {}
        for item in buf:
            agent_info[item[0]] = [item[1], item[2], item[3]]

        buf = np.empty((attack_event_ct, 3), dtype=np.int32)
        _LIB.env_get_info(
            self.game,
            -1,
            b"attack_event",
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        attack_event = buf

        return agent_info, attack_event

    def __del__(self):
        _LIB.env_delete_game(self.game)

    # ====== PRIVATE ======
    def _serialize_event_exp(self, config):
        """serialize event expression and sent them to game engine"""
        game = self.game

        # collect agent symbol
        symbol2int = {}
        config.symbol_ct = 0

        def collect_agent_symbol(node, config):
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_agent_symbol(item, config)
                elif isinstance(item, AgentSymbol):
                    if item not in symbol2int:
                        symbol2int[item] = config.symbol_ct
                        config.symbol_ct += 1

        for rule in config.reward_rules:
            on = rule[0]
            receiver = rule[1]
            for symbol in receiver:
                if symbol not in symbol2int:
                    symbol2int[symbol] = config.symbol_ct
                    config.symbol_ct += 1
            collect_agent_symbol(on, config)

        # collect event node
        event2int = {}
        config.node_ct = 0

        def collect_event_node(node, config):
            if node not in event2int:
                event2int[node] = config.node_ct
                config.node_ct += 1
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_event_node(item, config)

        for rule in config.reward_rules:
            collect_event_node(rule[0], config)

        # send to C++ engine
        for sym in symbol2int:
            no = symbol2int[sym]
            _LIB.gridworld_define_agent_symbol(game, no, sym.group, sym.index)

        for event in event2int:
            no = event2int[event]
            inputs = np.zeros_like(event.inputs, dtype=np.int32)
            for i, item in enumerate(event.inputs):
                if isinstance(item, EventNode):
                    inputs[i] = event2int[item]
                elif isinstance(item, AgentSymbol):
                    inputs[i] = symbol2int[item]
                else:
                    inputs[i] = item
            n_inputs = len(inputs)
            _LIB.gridworld_define_event_node(
                game, no, event.op, as_int32_c_array(inputs), n_inputs
            )

        for rule in config.reward_rules:
            # rule = [on, receiver, value, terminal]
            on = event2int[rule[0]]

            receiver = np.zeros_like(rule[1], dtype=np.int32)
            for i, item in enumerate(rule[1]):
                receiver[i] = symbol2int[item]
            if len(rule[2]) == 1 and rule[2][0] == "auto":
                value = np.zeros(receiver, dtype=np.float32)
            else:
                value = np.array(rule[2], dtype=np.float32)
            n_receiver = len(receiver)
            _LIB.gridworld_add_reward_rule(
                game,
                on,
                as_int32_c_array(receiver),
                as_float_c_array(value),
                n_receiver,
                rule[3],
            )


class CircleRange:
    def __init__(self, radius):
        """Define a circle range for attack or view

        Args:
            radius (float): Radius of vision around the agent.
        """
        self.radius = radius
        self.angle = 360

    def __str__(self):
        return "circle(%g)" % self.radius


class SectorRange:
    def __init__(self, radius, angle):
        """Define a sector range for attack or view.

        Args:
            radius (float): Radius of vision around the agent.
            angle (float): Angle (<180 degrees) describing the width of vision.
        """
        self.radius = radius
        self.angle = angle
        if self.angle >= 180:
            raise Exception("the angle of a sector should be smaller than 180 degree")

    def __str__(self):
        return f"sector({self.radius:g}, {self.angle:g})"
