# Import pygame
import pygame
import numpy
import tensorflow as tf
import keras

visited_positions = {}

def step(state, action):
    # action: 0 - влево, 1 - вправо, 2 - вверх, 3 - вниз
    if action == 0:
        state[0] -= 1
    elif action == 1:
        state[0] += 1
    elif action == 2:
        state[1] -= 1
    elif action == 3:
        state[1] += 1

    # ограничения
    state[0] = max(0, min(GOAL, state[0]))
    state[1] = max(0, min(GOAL, state[1]))

    # проверка завершения эпизода
    done = (state[0] == target_pos[0] and state[1] == target_pos[1])

    # награда: чем ближе к цели, тем больше
    distance = abs(target_pos[0] - state[0]) + abs(target_pos[1] - state[1])
    reward = max(0, GOAL*2 - distance)  # награда растет при уменьшении distance

    # если достиг цели, награда удваивается
    if done:
        reward *= 2

    # проверка повторов позиции
    pos_key = tuple(state)
    visited_positions[pos_key] = visited_positions.get(pos_key, 0) + 1
    if visited_positions[pos_key] == 3:
        reward /= 2 

    return state, reward, done


def pos_to_pix(pos):
    return (pos * 50)

def choose_action(state, epsilon=0.9):
    if numpy.random.rand() < epsilon:
        return numpy.random.randint(0, 4)
    q = model.predict(numpy.array([state]), verbose=0)  # <- исправлено
    return numpy.argmax(q)

def train_step(state):
    action = choose_action(state)
    next_state, reward, done = step(state, action)

    future_q = numpy.max(
        model.predict(numpy.array([next_state]), verbose=0)  # <- исправлено
    )

    target = reward
    if not done:
        target += GAMMA * future_q

    target_f = model.predict(numpy.array([state]), verbose=0)  # <- исправлено
    target_f[0][action] = target

    model.fit(numpy.array([state]), target_f, verbose=0)  # <- исправлено

    return next_state, done

pygame.init()

size = width,height = 600, 600
screen = pygame.display.set_mode(size)

clock = pygame.time.Clock()


AIcharacter = pygame.image.load('test.jpg')
DEFAULT_CHARACTER_SIZE = (100, 100)
DEFAULT_CHARACTER_POSITION = (250,250)
AIcharacter = pygame.transform.scale(AIcharacter, DEFAULT_CHARACTER_SIZE)

targetImage = pygame.image.load('circle.png')
DEFAULT_TARGET_SIZE = (50, 50)
DEFAULT_TARGET_POSITION = (50,250)
targetImage = pygame.transform.scale(targetImage, DEFAULT_CHARACTER_SIZE)

reward = -1

target_pos = [5, 2]
ai_pos = [0, 0]

STATE_SIZE = 2     # позиция
ACTION_SIZE = 4    # назад / вперёд
GOAL = 10
GAMMA = 0.95       # важность будущей награды
LR = 0.01
# STATE_SIZE	сколько чисел описывает состояние
# ACTION_SIZE	сколько вариантов действий
# GAMMA	"насколько я думаю о будущем"
# LR	насколько быстро нейросеть меняет веса


model = tf.keras.Sequential([
    keras.layers.Dense(24, activation="relu", input_shape=(STATE_SIZE,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(ACTION_SIZE, activation="linear")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="mse"
)

running = False
doneCounter = 0

while not running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = True

    ai_pos, done = train_step(ai_pos)
    if done:
        ai_pos = [0, 0]
        doneCounter += 1
        print(doneCounter)
    

    # render
    screen.fill((0, 0, 0))

    screen.blit(AIcharacter, (pos_to_pix(ai_pos[0]), pos_to_pix(ai_pos[1])))
    screen.blit(targetImage, (pos_to_pix(target_pos[0]), pos_to_pix(target_pos[1])))

    pygame.display.flip()
    clock.tick(10)

