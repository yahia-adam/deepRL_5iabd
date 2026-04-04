#include <vector>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

class QuartoEnv {
private:
    std::mt19937 rng;

    int rand_int(int max_val) {
        std::uniform_int_distribution<int> dist(0, max_val - 1);
        return dist(rng);
    }

public:
    vector<int> all_pieces;
    vector<vector<int>> victory_patterns;
    vector<int> board;
    vector<int> selected_piece;
    vector<int> available_pieces;

    mutable vector<int> observation_space_cache;
    mutable vector<int> action_space;
    
    bool is_game_over_flag;
    bool selecting;
    int  current_player;
    int  agent_player;
    int  pieces_left;
    float score;

    QuartoEnv(int seed = -1) {
        if (seed == -1) {
            std::random_device rd;
            rng.seed(rd());
        } else {
            rng.seed(seed);
        }

        observation_space_cache.assign(4 + 64 + 64, -1);
        action_space.assign(16 * 2, 0);

        victory_patterns = {
            {0, 1, 2, 3},    {4, 5, 6, 7},    {8, 9, 10, 11},  {12, 13, 14, 15},
            {0, 4, 8, 12},   {1, 5, 9, 13},   {2, 6, 10, 14},  {3, 7, 11, 15},
            {0, 5, 10, 15},  {3, 6, 9, 12}
        };

        all_pieces = {
            1, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 0,
            1, 1, 0, 1,  1, 0, 1, 0,  1, 0, 1, 1,  1, 1, 1, 0,
            0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  0, 1, 0, 0,
            0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 1, 1,  0, 1, 1, 0
        };

        reset();
    }

    void reset() {
        available_pieces = all_pieces;
        board.assign(64, -1);
        selected_piece.assign(4, -1);
        agent_player = rand_int(2);

        is_game_over_flag = false;
        selecting = true;
        current_player = 0;
        pieces_left = 16;
        score = 0.0f;
    }

    void step(int action) {
        if (is_game_over_flag) return;

        if (selecting) {
            action -= 16;
            int offset = action * 4;
            for (int a = 0; a < 4; a++) {
                selected_piece[a] = available_pieces[offset + a];
                available_pieces[offset + a] = -1;
            }
            current_player = (current_player == 1) ? 0 : 1;
        } else {
            int offset = action * 4;
            for (int a = 0; a < 4; a++) {
                board[offset + a] = selected_piece[a];
                selected_piece[a] = -1;
            }
            pieces_left--;

            for (const auto& pattern : victory_patterns) {
                int a = pattern[0], b = pattern[1], c = pattern[2], d = pattern[3];
                if (board[a * 4] == -1) continue;
                
                for (int i = 0; i < 4; i++) {
                    if (board[a*4+i] != -1 && 
                        board[a*4+i] == board[b*4+i] &&
                        board[b*4+i] == board[c*4+i] &&
                        board[c*4+i] == board[d*4+i]) 
                    {
                        is_game_over_flag = true;
                        score = (current_player == agent_player) ? 1.0f : -1.0f;
                        break; 
                    }
                }
                if (is_game_over_flag) break;
            }

            if (pieces_left == 0 && !is_game_over_flag) {
                is_game_over_flag = true;
                score = 0.0f; 
            }
        }
        selecting = !selecting; 
    }

    vector<int> get_action_mask() const {
        action_space.assign(16 * 2, 0);
        if (!selecting) {
            for (int i = 0; i < 16; ++i) {
                if (board[i * 4] == -1) action_space[i] = 1;
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                if (available_pieces[i * 4] != -1) action_space[16 + i] = 1;
            }
        }
        return action_space;
    }

    vector<int> get_observation_mask() const {
        int n = 0;
        for (int i = 0; i < 4; i++, n++) {
            observation_space_cache[n] = selected_piece[i];
        }
        for (int i = 0; i < 64; i++, n++) {
            observation_space_cache[n] = board[i];
        }
        for (int i = 0; i < 16; i++, n++) {
            observation_space_cache[n] = available_pieces[i];
        }
        return observation_space_cache;
    }
};

// --- WRAPPER PYBIND11 ---
PYBIND11_MODULE(quarto_cpp, m) {
    py::class_<QuartoEnv>(m, "QuartoEnvCpp")
        .def(py::init<int>(), py::arg("seed") = -1)
        .def("reset", &QuartoEnv::reset)
        .def("step", &QuartoEnv::step)
        .def("get_action_mask", &QuartoEnv::get_action_mask)
        .def("get_observation_mask", &QuartoEnv::get_observation_mask)
        
        // Exposer les variables publiques à Python
        .def_readwrite("board", &QuartoEnv::board)
        .def_readwrite("selected_piece", &QuartoEnv::selected_piece)
        .def_readwrite("available_pieces", &QuartoEnv::available_pieces)
        .def_readwrite("is_game_over", &QuartoEnv::is_game_over_flag)
        .def_readwrite("selecting", &QuartoEnv::selecting)
        .def_readwrite("current_player", &QuartoEnv::current_player)
        .def_readwrite("agent_player", &QuartoEnv::agent_player)
        .def_readwrite("score", &QuartoEnv::score);
}