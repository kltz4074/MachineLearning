import pygame
import numpy
import tensorflow as tf
import keras
from collections import deque
import random
import os

visited_positions = {}

target_pos = [5, 2]
ai_pos = [0, 0]

TOTAL_EPOCHS = 50           # сколько всего эпох хотим
EPISODES_PER_EPOCH = 50     # сколько эпизодов в одной эпохе
MAX_STEPS_PER_EPISODE = 100 # ограничение шагов в одном эпизоде (защита от зацикливания)

STATE_SIZE = 2     # 2D позиция (x, y)
ACTION_SIZE = 4    # назад / вперёд / вверх / вниз
GAMMA = 0.95       # важность будущей награды
LR = 0.0005        # насколько быстро нейросеть меняет веса

GOAL = 10 

epsilon = 1.0           # шанс на рандомные движения нейросети (с временем стаёт меньше)
epsilon_min = 0.00      # минимальное значение epsilon
epsilon_decay = 0.9992  # скорость изменения epsilon

FitModel = False # если уже обученая модель загружена то нужно её дообучивать? 

SAVE_DIR = "saved_models" # папка куда сохраняются модели
MODEL_NAME = "Walking_qq" # название модели
CHECKPOINT_EVERY = 5      # каждые сколько эпох сохранять модель?


# просто обьявляем модель
model = None 
target_model = None


os.makedirs(SAVE_DIR, exist_ok=True)

def save_agent(epoch, model, target_model=None): # сохраняем модель
    epoch_str = f"{epoch:04d}"
    base_name = f"{MODEL_NAME}_epoch_{epoch_str}"
    
    full_path = os.path.join(SAVE_DIR, f"{base_name}.keras")
    
    keras.saving.save_model(model, full_path)
    print(f"Сохранена полная модель: {full_path}")
    
    state_path = os.path.join(SAVE_DIR, f"{base_name}_state.txt")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\n")
        f.write(f"step_count={step_count}\n")
        f.write(f"epsilon={epsilon:.6f}\n")
    
    print(f"Состояние обучения сохранено: {state_path}")

def load_last_model():
    """пробуем загрузить самую последнюю сохранённую модель .keras"""
    global model, target_model, epsilon, step_count
    
    # Ищем все .keras файлы, которые начинаются с MODEL_NAME
    model_files = [
        f for f in os.listdir(SAVE_DIR) 
        if f.startswith(MODEL_NAME) and f.endswith(".keras")
    ]
    
    if not model_files:
        print("Сохранённых моделей не найдено → создаём новую")
        return False
    
    # Берём самую новую по номеру эпохи в имени
    latest_file = max(model_files, key=lambda x: int(x.split("_epoch_")[1].split(".")[0]))
    model_path = os.path.join(SAVE_DIR, latest_file)
    
    print(f"Нашлась модель: {model_path}")
    print("Загружаю... ", end="")
    
    # Загружаем полную модель
    model = keras.saving.load_model(model_path)
    
    # Создаём target-модель как копию
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    
    # Пытаемся восстановить epsilon и step_count
    state_file = model_path.replace(".keras", "_state.txt")
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("epsilon="):
                    epsilon = float(line.split("=", 1)[1])
                if line.startswith("step_count="):
                    step_count = int(line.split("=", 1)[1])
        print(f"восстановлено! epsilon = {epsilon:.3f}, step_count = {step_count}")
    else:
        print("но файл состояния не найден (epsilon и step_count останутся текущими)")
    
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

    done = (state[0] == target_pos[0] and state[1] == target_pos[1])
    # награда: чем ближе к цели, тем больше
    distance = abs(target_pos[0] - state[0]) + abs(target_pos[1] - state[1])
    reward = max(0, GOAL*2 - distance)  # награда растет при уменьшении distance
    if done:
        reward *= 2
    return state, reward, done


def pos_to_pix(pos):
    return (pos * 50)


def choose_action(state):
    global epsilon
    
    state_array = numpy.array(state)  
    
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
        
        q_next = model.predict(next_states, verbose=0)
        q_target_next = target_model.predict(next_states, verbose=0)
        
        targets = rewards + GAMMA * q_target_next[numpy.arange(BATCH_SIZE), numpy.argmax(q_next, axis=1)] * (1 - dones)
        
        target_f = model.predict(states, verbose=0)
        target_f[numpy.arange(BATCH_SIZE), actions] = targets
        
        if loaded == False & FitModel == True:
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



print("Попытка загрузки последней модели...")
loaded = load_last_model()

if loaded:
    print("Модель успешно загружена и готова к использованию\n")
else:
    print("Модель не найдена → создаём новую\n")
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
        # ресетаем статус эпизода
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

            # рендер
            screen.fill((0, 0, 0))
            screen.blit(AIcharacter, (pos_to_pix(ai_pos[0]), pos_to_pix(ai_pos[1])))
            screen.blit(targetImage, (pos_to_pix(target_pos[0]), pos_to_pix(target_pos[1])))
            pygame.display.flip()
            clock.tick(100)

        # если максимум шагов достигнут то заканчиваем эпизод
        if not episode_done:
            current_episode += 1
            print(f"episode {current_episode:3d}/{EPISODES_PER_EPOCH}  "
                    f"triggered step limit ({steps_in_episode})  ε: {epsilon:.3f}")
    
    # конец эпохи
    current_epoch += 1
    if (current_epoch + 1) % CHECKPOINT_EVERY == 0 & FitModel == True:
        save_agent(current_epoch, model, target_model)
