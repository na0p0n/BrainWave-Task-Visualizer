import sys, random
import os
import pygame
import time
from pygame.locals import *
import nouha_recv as bwr
from socket import socket, AF_INET, SOCK_DGRAM

import TaskManager

#”重要” offLineModeをFalseにすると脳波データの通信が行われるため、UDP通信が起動していない場合にFalseにするとバグ発生の可能性あり
offLineMode = True
WIDTH = 3200        #ウィンドウの横サイズ
HEIGHT = 1800       #ウィンドウの縦サイズ
TIME = 30000        #タスクの継続時間（ミリ秒）
BREAK_TIME = 15000  #タスク間の休憩時間（ミリ秒）
PREP_TIME = 10000   #各タスクの「慣れ」時間（ミリ秒）
TASK_COUNT = 10      #右、左、ニュートラルそれぞれのタスク数（デフォルトの場合それぞれ5回ずつタスクを実施）
#UDPの準備
HOST = ''
PORT = 8001
#サウンドファイルの指定
SOUND_DIR = "sounds"
BLINK_INTERVAL = 1000
white = (255, 255, 255)
black = (0, 0, 0)
screen_type = ["sound", "screen", "task"]
text_type = {"sound":"音声再生・脳波測定中…", "screen":"画面指示・脳波測定中…", "task":"タスク指示・脳波測定中…"}
mind_type = ["right", "left", "neutral"]
musics = ["001", "002", "003", "420"]
position_rect = (0, 0, 0, 0)
position_circle = (0, 0)
get_tick_time = [0, 0] #[before, now]
# 指示内容を定義
instructions = {
    "right": [

        "右手を想像して箸で食べ物を掴んでください。",
        "右手のボタンを押してください。",
        "右手のレバーを操作してください。",
    ],
    "left": [
        
        "左手を想像して箸で食べ物を掴んでください。",
        "左手のボタンを押してください。",
        "左手のレバーを操作してください。",
    ],
    "neutral": [
        "目を閉じてリラックスしてください。",
        "深呼吸をしてください。",
        "座禅を組んで瞑想してください。"
    ]
}

#アプリの初期化
def initialize_pygame():
    try:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        get_tick_time = [pygame.time.get_ticks(), pygame.time.get_ticks()]
        return screen
    except pygame.error as e:
        print(f"Pygameの初期化エラー： \n{e}")
        sys.exit(1)

#音声・音楽の再生
def load_and_play_sound(filename):
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except pygame.error as e:
        print(f"音楽ファイル{filename}の読み込みエラー： \n{e}")
        return False
    return True

#脳波データの受け取りと処理
def send_nouhadata(s, choice, mind, process):
    print_text = text_type[choice]
    print("タスク経過時間：", str(get_tick_time[1] - get_tick_time[0]), print_text) 
    if offLineMode == False and get_tick_time[1] - get_tick_time[0] >= PREP_TIME:
        data, address = s.recvfrom(1024)
        process.Receive_BrainWave(nouha=data, key=mind, address=address)
    get_tick_time[1] = pygame.time.get_ticks()

#アプリの終了時の処理
def check_exit(s, choice, process):
    for event in pygame.event.get():
        if event.type == QUIT:
            try:
                if not offLineMode:
                    send_nouhadata(s, choice, mind="quit", process=process)
                pygame.quit()
                s.close()
                sys.exit()
            except Exception as e:
                print(f"終了処理中にエラー発生： \n{e}")
                sys.exit(1)
    return False

#メイン
def main():
    try:
        s = socket(AF_INET, SOCK_DGRAM)
        try:
            s.bind((HOST, PORT))
        except OSError as e:
            print(f"ポート{PORT}のバインドに失敗。ポートが既に開かれている可能性があります。： \n{e}")
            return
        screen = initialize_pygame()
        process = bwr.BrainWave_Receive()
        task_manager = TaskManager.TaskManager(TASK_COUNT)
        
        while True:
            try:
                get_tick_time[0] = pygame.time.get_ticks()
                mind = task_manager.get_next_type("mind")
                if mind is None:
                    print("すべてのタスクが完了しました。ウィンドウを閉じて終了してください。")
                    while True:
                        if check_exit(s, choice, process):
                            break
                choice = task_manager.get_next_type("task")
                if choice != "task" and mind == "neutral":
                    task_manager.sub_counts(mind, choice, 1)
                    continue
                
                if choice == "screen" and mind != "neutral":
                    try:
                        load_and_play_sound(os.path.join(SOUND_DIR, "画面を注視してください.wav"))
                        last_blink = pygame.time.get_ticks()
                        is_visible = True
                        screen.fill(black)
                        #円の描画
                        if mind == "right":
                            position_rect = (WIDTH / 2, 0, WIDTH, HEIGHT)
                            position_circle = (WIDTH * 3 / 4, HEIGHT / 2)
                        elif mind == "left":
                            position_rect = (0, 0, WIDTH / 2, HEIGHT)
                            position_circle = (WIDTH / 4, HEIGHT / 2)
                        
                        while get_tick_time[1] - get_tick_time[0] <= TIME:
                            current_time = pygame.time.get_ticks()
                            
                            if current_time - last_blink >= BLINK_INTERVAL:
                                is_visible = not is_visible
                                last_blink = current_time
                                if is_visible:
                                    screen.fill(black)
                                    screen.fill((255, 255, 0), position_rect)
                                    pygame.draw.circle(screen, (255, 0, 255), position_circle, 150)
                                else:
                                    screen.fill(black)
                                pygame.display.update()
                            send_nouhadata(s, choice, mind, process)
                            if check_exit(s, choice, process):
                                break
                        screen.fill(black)
                            
                    except pygame.error as e:
                        print(f"画面描画エラー： \n{e}")
                        
                elif choice == "sound" and mind != "neutral":
                    screen.fill(black)
                    music_choice = random.choice(musics)
                    filename = os.path.join(SOUND_DIR, music_choice+"_"+mind+".mp3")
                    if not load_and_play_sound(filename):
                        continue
                    while pygame.mixer.music.get_busy():
                        send_nouhadata(s, choice, mind, process)
                        if check_exit(s, choice, process):
                            break
                    screen.fill(black)
                            
                elif choice == "task":
                    screen.fill(black)
                    instruction = random.choice(instructions[mind])
                    filename = os.path.join(SOUND_DIR, mind+"_"+instruction+".wav")
                    load_and_play_sound(filename)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {instruction}")
                    while get_tick_time[1] - get_tick_time[0] <= TIME:
                        send_nouhadata(s, choice, mind, process)
                        if check_exit(s, choice, process):
                            break
                    load_and_play_sound(filename = os.path.join(SOUND_DIR, "タスクを終了してください.wav"))
                    screen.fill(black)
                
                #画面の更新
                pygame.display.update()
                
                #タスク間の休憩
                get_tick_time[0] = pygame.time.get_ticks()
                print(f"現在の実行回数: {task_manager.get_counts()}\nタスク実行回数: {task_manager.get_taskCounts()}")
                while get_tick_time[1] - get_tick_time[0] <= BREAK_TIME:
                    if check_exit(s, choice, process):
                        break
                    get_tick_time[1] = pygame.time.get_ticks()
            except SystemExit:
                pygame.quit()
                s.close()
                sys.exit()
            except Exception as e:
                print(f"予期せぬエラー： \n{e}")
                continue
    except Exception as e:
        print(f"重大なエラー： \n{e}")
    finally:
        try:
            pygame.quit()
            s.close()
        except Exception as e:
            print(f"終了処理中にエラーが発生しました：\n{e}")
            sys.exit(1)

#直接起動時の処理
if __name__ == "__main__":
    main()
        
# process = bwr.BrainWave_Receive #インスタンス作成
# while True:
#     timeClock.tick(250)
#     mind = random.choice(mind_type)
#     choice = random.choice(screen_type)
    
#     if choice == "screen" and mind != "neutral":
#         screen.fill(black)
#         #円の描画
#         if mind == "right":
#             position_rect = (width / 2, 0, width, height)
#             position_circle = (width * 3 / 4, height / 2)
#         elif mind == "left":
#             position_rect = (0, 0, width / 2, height)
#             position_circle = (width / 4, height / 2)
#         screen.fill((255, 255, 0), position_rect)
#         pygame.draw.circle(screen, (255, 0, 255), position_circle, 150)
#         pygame.display.update()
#         while timeClock.get_time() <= 60000:
#             print(str(timeClock.get_time()))
#             if offLineMode == False:
#                 data, address = s.recvfrom(1024)
#                 process.Receive_BrainWave(nouha=data, key=mind, address=address)
#     elif choice == "sound" and mind != "neutral":
#         screen.fill(black)
#         if mind == "right":
#             pygame.mixer.music.load(filename_right)
#             pygame.mixer.music.rewind()
#         elif mind == "left":
#             pygame.mixer.music.load(filename_left)
#             pygame.mixer.music.rewind()
#         while pygame.mixer.music.get_busy():
#             print("Playing...")
#             if offLineMode == False:
#                 data, address = s.recvfrom(1024)
#                 process.Receive_BrainWave(nouha=data, key=mind, address=address)
#     elif choice == "task":
#         screen.fill(black)
#         instruction = random.choice(instructions[mind])
#         filename = ".\\" + mind + "_" + instruction + ".wav"
#         pygame.mixer.music.load(filename)
#         pygame.mixer.music.play()
#         text = instruction
#         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {instruction}")
#         while pygame.mixer.music.get_busy():
#             print("Playing...")
#         while timeClock.get_time() <= 60000:
#             print(str(timeClock.get_time()))
#             if offLineMode == False:
#                 data, address = s.recvfrom(1024)
#                 process.Receive_BrainWave(nouha=data, key=mind, address=address)
        
#     #画面の更新
#     pygame.display.update()
#     pygame.time.wait(10000)     #10秒間クールダウン
    
#     #終了イベント
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             process.Receive_BrainWave(nouha=data, key=9, address=address)
#             pygame.quit()
#             sys.exit()
            