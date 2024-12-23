# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:33:19 2024

@author: Naoya

FocusCalmから得たデータをcsv化するプログラム
BandPower + Attention + Meditationの同時取得可能
それぞれ受信したデータが無い場合にはファイル作成を飛ばす

2024/12/12　更新
・外部からデータを記録したいとき用にクラス化

"""
from __future__ import unicode_literals, print_function
from socket import socket, AF_INET, SOCK_DGRAM

import numpy as np
import matplotlib.pyplot as plt
import struct
import csv
import keyboard
import datetime
from pythonosc import osc_message
import sys
import collections
import os

waves = []
attentions = []
meditations = []
keys = []
key_type = {"neutral":0, "left":1, "right":2, "break":5,"quit":9}
SAVE_DIR = "csvfiles"

#
#EEGから送られたデータを配列に変換する関数
#引数：data（EEGデータ）
#返り値：リスト[δ波、θ波、α波、β波、γ波]
def Convert_BrainWave(data):
    try:
        msg = osc_message.OscMessage(data)
        # print(msg.params)
        types = msg.address
        arguments = []
        if  types == "/Attention":
            arguments.append("Attention")
            arguments.append(float(msg.params[0]))
            
        elif types == "/Meditation":
            arguments.append("Meditation")
            arguments.append(float(msg.params[0]))
            
        elif types == "/BandPower":
            arguments.append("BandPower")
            arguments += list(map(float, msg.params[0].split(";")))
        else:
            raise ValueError(f"未知のデータ形式： {msg.address}")
        return arguments
    except osc_message.ParseError:
        print("OSCメッセージのパースに失敗")
        return None
    except ValueError as e:
        print(f"データ変換エラー： \n{e}")
        return None
    except Exception as e:
        print(f"予期せぬエラー： \n{e}")
        return None
    
class BrainWave_Receive:
    def __init__(self):
        try:
            self.waves = []
            self.attentions = []
            self.meditations = []
            self.keys = []
        except Exception as e:
            print(f"初期化エラー： \n{e}")
            raise
        
    def Receive_BrainWave(self, nouha, key, address):
        key = key_type[key]
        if key == 9:
            self.SaveFile()
            return
        elif key == 5:
            converted_datas = ["Break", "", "", "", "", ""]
        else:
            converted_datas = Convert_BrainWave(nouha)
            converted_datas.append(key)
        # 受信するデータによっていれる配列を変える
        

        if converted_datas[0] == "BandPower":
            self.waves.append(converted_datas[1:])
            print(f"\n type: {converted_datas[0]}\n alpha: {converted_datas[1]}\n beta: {converted_datas[2]}\n theta: {converted_datas[3]}\n delta: {converted_datas[4]}\n gamma: {converted_datas[5]} \n key: {converted_datas[6]}\n from: {address}\n")
            keys = [row[5] for row in self.waves]
        elif converted_datas[0] == "Attention":
            self.attentions.append(converted_datas[1:])
            print(f"\n type: {converted_datas[0]}\n Attention: {converted_datas[1]}\n key: {converted_datas[2]}\n from: {address}\n")
        elif converted_datas[0] == "Meditation":
            self.meditations.append(converted_datas[1:])
            print(f"\n type: {converted_datas[0]}\n Meditation: {converted_datas[1]}\n key: {converted_datas[2]}\n from: {address}\n")
        elif converted_datas[0] == "Break":
            self.waves.append(["", "", "", "", ""])
            self.attentions.append(["", ""])
            self.meditations.append(["", ""])

    def SaveFile(self):
        #csvファイルの作成
        try:
            timestamp = str(datetime.datetime.now()).replace(" ", ",").replace(".", "-").replace(":", "-")
            for data_type, data_list, header in [
                ("BandPower", self.waves, ["alpha", "beta", "theta", "delta", "gamma", "key"]),
                ("Attention", self.attentions, ["attention", "key"]),
                ("Meditation", self.meditations, ["meditation", "key"])
            ]:
                if data_list:
                    filename = os.path.join(SAVE_DIR, f'nouhadata_{data_type}{timestamp}.csv')
                    try:
                        with open(filename, 'w', newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(header)
                            writer.writerows(data_list)
                            print(f"csvファイルの生成に成功しました。ファイル名：{filename}")
                    except PermissionError:
                        print(f"ファイル{filename}への書き込み権限がありません。")
                    except Exception as e:
                        print(f"ファイルへの書き込みエラー。\n{e}")
        except Exception as e:
            print(f"保存処理のエラー： \n{e}")

        # if not len(self.waves) == 0:
        #     with open(r'.\nouhadata_BandPower' + timestamp + ".csv", 'w', newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["alpha","beta","theta","delta","gamma","key"])
        #         writer.writerows(self.waves)
        #         print(f"csvファイルを生成しました。ファイル名：nouhadata_BandPower{timestamp}.csv")

        # if not len(self.attentions) == 0:
        #     with open(r'.\nouhadata_Attention' + timestamp + ".csv", 'w', newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["attention","key"])
        #         writer.writerows(self.attentions)
        #         print(f"csvファイルを生成しました。ファイル名：nouhadata_Attention{timestamp}.csv")
                
        # if not len(self.meditations) == 0:
        #     with open(r'.\nouhadata_Meditation' + timestamp + ".csv", 'w', newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["meditation","key"])
        #         writer.writerows(self.meditations)
        #         print(f"csvファイルを生成しました。ファイル名：nouhadata_Meditation{timestamp}.csv")

#このプログラムを直接開いた時の処理
def main(process):
    while True:
        data, address = s.recvfrom(1024)
        if keyboard.is_pressed('F8'):
            recvFlag = (recvFlag + 1) % 2

        if recvFlag == 1:
            if keyboard.is_pressed('left') or keyboard.is_pressed('a'): #左キーを押したとき
                key = 1
            elif keyboard.is_pressed('right') or keyboard.is_pressed('d'): #右キーを押したとき
                key = 2
            elif keyboard.is_pressed('up') or keyboard.is_pressed('w'): #右キーを押したとき
                key = 0
            elif keyboard.is_pressed('q'):
                key = 9
                process.Receive_BrainWave(data, key, address)
                print("処理を中止します")
                s.close()
                break
            else:
                pass
            process.Receive_BrainWave(data, key)#終了しないときはここに到達
        else:
            if keyboard.is_pressed('q'):
                print("処理を中止します")
                s.close()
                break
            print("recvFlag = ", recvFlag, ", key = ", key)
            if len(waves) > 0 and (1 in keys or 2 in keys):
                print(collections.Counter(keys))
        

    #処理が最後までに終了したあとにポートを閉じる
    s.close()
    saveFlag = int(input("保存しますか？　保存する：1、保存しない：2 >>"))

    if saveFlag == 2:
        sys.exit()
    process.SaveFile()
        

#メインの処理（このファイルを直接開いたとき）
#データを受け取って処理、キー入力を認識してデータに追加
#何も押さなければkey=0、左キーでkey=1、右キーでkey=2、上キーでkey=3、下キーでkey=4、qキーで処理の中止
#最新のデータのみ毎回表示、
if __name__ == "__main__":
    HOST = ''
    PORT = 8001
    s = socket(AF_INET, SOCK_DGRAM)
    s.bind((HOST, PORT))
    recvFlag = 0
    key = 0
    process = BrainWave_Receive()
    main(process)

