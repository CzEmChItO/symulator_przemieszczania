import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Tworzenie planszy 2D.
def create_board(size, p, start, end):
    board = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if np.random.rand() < p:
                board[i][j] = -1
    board[start] = 0
    board[end] = 0
    return board


# funkcja rysująca siatkę miasta z krawędziami i zaznaczonym punktem początkowym i końcowym
def draw_city(start, end, board, path=None):
    L = len(board)
    fig, ax = plt.subplots()
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_yticklabels(reversed(ax.get_yticks()))
    ax.xaxis.set_ticks_position('top')
    ax.plot([0, L-1], [0, 0], color='black')
    ax.plot([0, L-1], [L-1, L-1], color='black')
    ax.plot([0, 0], [0, L-1], color='black')
    ax.plot([L-1, L-1], [0, L-1], color='black')
    for i in range(L):
        for j in range(L):
            if board[i][j] == -1:
                ax.scatter(j, L - i - 1, color='black', s=100, marker='o')
    ax.scatter(start[1], L - start[0] - 1, label='START', color='green', s=100, marker='o')
    ax.scatter(end[1], L - end[0] - 1, label='END', color='red', s=100, marker='o')
    if path is not None:
        line, = ax.plot([], [], color='yellow')
        x_vals = [start[1]]
        y_vals = [L - start[0] - 1]
        for point in path:
            x_vals.append(point[1])
            y_vals.append(L - point[0] - 1)

        def update(frame):
            if frame >= len(x_vals):
                return
            line.set_data(x_vals[:frame + 1], y_vals[:frame + 1])
            return line,
        anim = FuncAnimation(fig, update, frames=len(x_vals), interval=500, blit=True)
    else:
        ax.text(0.5, 0.5, 'BRAK SCIEŻKI', ha='left', va='center', fontsize=20)
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left')
    ax.set_title('Symulator przejazdu od punku do punktu z uwzglednieniem blokad')
    plt.show()


def dijkstra(start, end, board):
    # Inicjalizacja listy odległości do każdego punktu na planszy.
    distances = {(i, j): float('inf') for i in range(board.shape[0]) for j in range(board.shape[1])}
    distances[start] = 0
    # Inicjalizacja listy poprzedników dla każdego punktu na planszy.
    predecessors = {start: None}
    # Inicjalizacja kolejki priorytetowej.
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        # Jeśli dotarliśmy do końca, zwracamy znalezioną drogę.
        if current_vertex == end:
            path = []
            while current_vertex is not None:
                path.append(current_vertex)
                current_vertex = predecessors[current_vertex]
            return path[::-1]
        # W przeciwnym wypadku, przeglądamy sąsiadów aktualnego punktu.
        for neighbor in get_neighbors(current_vertex, board):
            distance = current_distance + 1  # koszt przejścia to jedna jednostka czasu
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    # Jeśli nie znaleziono drogi do końca, zwracamy None.
    return None


def get_neighbors(vertex, board):
    neighbors = []
    x, y = vertex
    if x > 0 and board[x - 1, y] != -1:
        neighbors.append((x - 1, y))
    if x < board.shape[0] - 1 and board[x + 1, y] != -1:
        neighbors.append((x + 1, y))
    if y > 0 and board[x, y - 1] != -1:
        neighbors.append((x, y - 1))
    if y < board.shape[1] - 1 and board[x, y + 1] != -1:
        neighbors.append((x, y + 1))
    return neighbors


def find_coords_of_negatives(board):
    L = len(board)
    coords = []
    for i in range(L):
        for j in range(L):
            if board[i][j] == -1:
                coords.append((i, j))
    return coords


# Parametry symulacji
L = 10  # Wielkość siatki miasta
p = 0.2  # Prawdopodobieństwo blokady ulicy

# Losowanie punktów startowego i końcowego
start = (np.random.randint(0, L), np.random.randint(0, L))
end = (np.random.randint(0, L), np.random.randint(0, L))
while start == end:
    end = (np.random.randint(0, L), np.random.randint(0, L))

board = create_board(L, p, start, end)
path = dijkstra(start, end, board)
if path is None:
    print("BRAK DROGI")

# Wypisanie podstawowych informacji
print(f'Rozmiar planszy: {L}x{L}')
print(f'Wspolrzedne punktu START: {start}')
print(f'Wspolrzedne punktu END: {end}')
print(f'Lista zablokowanych krzyzowan: {find_coords_of_negatives(board)}')
if path is not None:
    print(f'Koszt przejscia: {len(path)-1} jednostek czasu')
else:
    print("Brak sciezki laczacej punkt startu i konca.")
print(f'\nPlansza:\n{board}\n')

# narysuj siatkę miasta z krawędziami i punktami startu i końca
draw_city(start, end, board, path)
