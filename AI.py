import pygame
import numpy
import tensorflow as tf
import keras
from collections import deque
import random
import matplotlib as plt
import os


visited_positions = {}

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "Walking_qq"
CHECKPOINT_EVERY = 5   # сохранять каждые N эпох


def save_agent(epoch):
    path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_epoch_{epoch:04d}")
    
    # Сохраняем основную модель
    model.save(f"{path}_model.h5")
    
    # Сохраняем target-модель (иногда полезно)
    # target_model.save(f"{path}_target.h5")
    
    # Можно также сохранить epsilon, step_count и т.д.
    with open(f"{path}_state.txt", "w") as f:
        f.write(f"epoch={epoch}\n")
        f.write(f"step_count={step_count}\n")
        f.write(f"epsilon={epsilon:.6f}\n")
    
    print(f"Модель сохранена: {path}_model.h5")


def load_agent(try_epoch=None):
    global model, target_model, step_count, epsilon
    
    if try_epoch is None:
        # ищем самую последнюю модель
        files = [f for f in os.listdir(SAVE_DIR) if f.endswith("_model.h5")]
        if not files:
            print("Сохранённых моделей не найдено, начинаем с начала")
            return False
        latest = max(files, key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))
        load_path = os.path.join(SAVE_DIR, latest)
    else:
        load_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_epoch_{try_epoch:04d}_model.h5")
        if not os.path.exists(load_path):
            print(f"Модель эпохи {try_epoch} не найдена")
            return False
    
    print(f"Загружаем модель: {load_path}")
    
    model = tf.keras.models.load_model(load_path)
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    
    # Можно также загрузить epsilon и step_count из state.txt
    state_file = load_path.replace("_model.h5", "_state.txt")
    if os.path.exists(state_file):
        with open(state_file) as f:
            for line in f:
                if line.startswith("epsilon="):
                    epsilon = float(line.split("=")[1])
                if line.startswith("step_count="):
                    step_count = int(line.split("=")[1])
    
    return True

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
    return state, reward, done


def pos_to_pix(pos):
    return (pos * 50)


# Внутри train_step меняем choose_action на:
def choose_action(state):
    global epsilon
    
    state_array = numpy.array(state)           # ← вот это главное
    
    if numpy.random.rand() < epsilon:
        action = numpy.random.randint(0, 4)
    else:
        q = model.predict(state_array.reshape(1, -1), verbose=0)[0]
        action = numpy.argmax(q)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return action

step_count = 0

def train_step(state):
    global step_count
    
    action = choose_action(state)
    next_state, reward, done = step(state.copy(), action)  # ← copy очень важен!
    
    remember(state, action, reward, next_state, done)
    
    if len(memory) > BATCH_SIZE:
        minibatch = random.sample(memory, BATCH_SIZE)
        
        states = numpy.array([t[0] for t in minibatch])
        actions = numpy.array([t[1] for t in minibatch])
        rewards = numpy.array([t[2] for t in minibatch])
        next_states = numpy.array([t[3] for t in minibatch])
        dones = numpy.array([t[4] for t in minibatch])
        
        # Double DQN вариант (ещё стабильнее)
        q_next = model.predict(next_states, verbose=0)
        q_target_next = target_model.predict(next_states, verbose=0)
        
        targets = rewards + GAMMA * q_target_next[numpy.arange(BATCH_SIZE), numpy.argmax(q_next, axis=1)] * (1 - dones)
        
        target_f = model.predict(states, verbose=0)
        target_f[numpy.arange(BATCH_SIZE), actions] = targets
        
        model.fit(states, target_f, epochs=1, verbose=0)
    
    step_count += 1
    if step_count % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(model.get_weights())
    
    return next_state, done


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


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


MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000
current_epoch = 0
memory = deque(maxlen=MEMORY_SIZE)

target_pos = [5, 2]
ai_pos = [0, 0]

TOTAL_EPOCHS = 50           # сколько всего эпох хотим
EPISODES_PER_EPOCH = 50     # сколько эпизодов в одной эпохе
MAX_STEPS_PER_EPISODE = 500 # ограничение шагов в одном эпизоде (защита от зацикливания)


STATE_SIZE = 2     # позиция
ACTION_SIZE = 4    # назад / вперёд
GOAL = 10
GAMMA = 0.95       # важность будущей награды
LR = 0.01
# STATE_SIZE	сколько чисел описывает состояние
# ACTION_SIZE	сколько вариантов действий
# GAMMA	"насколько я думаю о будущем"
# LR	насколько быстро нейросеть меняет веса

# Добавляем эти переменные
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9992     # довольно плавное затухание

model = tf.keras.Sequential([
    keras.layers.Dense(24, activation="relu", input_shape=(STATE_SIZE,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(ACTION_SIZE, activation="linear")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="mse"
)

target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())


running = False
doneCounter = 0
while current_epoch < TOTAL_EPOCHS and not running:
    current_episode = 0
    epoch_steps = 0
    
    print(f"\n=== epoch {current_epoch + 1}/{TOTAL_EPOCHS} ===")
    
    while current_episode < EPISODES_PER_EPOCH and not running:
        # reset episode status
        ai_pos = [0, 0]
        steps_in_episode = 0
        episode_done = False
        
        while not episode_done and steps_in_episode < MAX_STEPS_PER_EPISODE and not running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = True
                    break

            ai_pos, done = train_step(ai_pos)
            steps_in_episode += 1
            step_count += 1
            epoch_steps += 1

            if done:
                episode_done = True
                current_episode += 1
                print(f"Эпизод {current_episode:3d}/{EPISODES_PER_EPOCH}  "
                        f"шагов: {steps_in_episode:4d}  "
                        f"ε: {epsilon:.3f}")

            # отрисовка
            screen.fill((0, 0, 0))
            screen.blit(AIcharacter, (pos_to_pix(ai_pos[0]), pos_to_pix(ai_pos[1])))
            screen.blit(targetImage, (pos_to_pix(target_pos[0]), pos_to_pix(target_pos[1])))
            pygame.display.flip()
            clock.tick(100)

        # если вышли по максимальному количеству шагов — тоже считаем эпизод завершённым
        if not episode_done:
            current_episode += 1
            print(f"episode {current_episode:3d}/{EPISODES_PER_EPOCH}  "
                    f"triggered step limit ({steps_in_episode})  ε: {epsilon:.3f}")
    
    # конец эпохи
    current_epoch += 1
    if (current_epoch + 1) % 2 == 0:
        target_model.save(f"model_epoch_{current_epoch+1:03d}.h5")
        print(f"Сохранено: model_epoch_{current_epoch+1:03d}.h5")

        model.save(f"model_epoch_{current_epoch+1:03d}.h5")
        print(f"Сохранено: model_epoch_{current_epoch+1:03d}.h5")
