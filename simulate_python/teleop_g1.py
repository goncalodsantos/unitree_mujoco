import time
import math
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

if __name__ == "__main__":
    # 1. Ligar ao Domain ID 1 e à interface local
    ChannelFactoryInitialize(1, "lo")

    # 2. Configurar o Publisher para o lowcmd do G1
    low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_puber.Init()

    # Criar a estrutura de comando padrão para os 35 motores do G1
    cmd = unitree_hg_msg_dds__LowCmd_()
    
    # 1. Configuração base de rigidez para todas as juntas
    for i in range(35):
        cmd.motor_cmd[i].mode = 10   # Modo Servo Posição
        cmd.motor_cmd[i].kp = 150.0  # Aumentar drasticamente para as pernas não cederem!
        cmd.motor_cmd[i].kd = 8.0    # Amortecimento proporcional elevado
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0

   # 2. Injetar ângulos padrão para criar uma pose de agachamento estável (G1 Map)
    # Ângulos calculados para manter o CoM centrado nos pés (em radianos)
    hip_pitch_target = 0.3     # Inclina a coxa para a frente
    knee_target = 0.6          # Flete o joelho
    ankle_pitch_target = -0.3  # Sincroniza o tornozelo para estabilizar a planta do pé

    # --- PERNA ESQUERDA (Índices 0 a 5) ---
    cmd.motor_cmd[0].q = hip_pitch_target      # left_hip_pitch
    cmd.motor_cmd[1].q = 0.0                   # left_hip_roll
    cmd.motor_cmd[2].q = 0.0                   # left_hip_yaw
    cmd.motor_cmd[3].q = knee_target           # left_knee
    cmd.motor_cmd[4].q = ankle_pitch_target    # left_ankle_pitch
    cmd.motor_cmd[5].q = 0.0                   # left_ankle_roll

    # --- PERNA DIREITA (Índices 6 a 11) ---
    cmd.motor_cmd[6].q = hip_pitch_target      # right_hip_pitch
    cmd.motor_cmd[7].q = 0.0                   # right_hip_roll
    cmd.motor_cmd[8].q = 0.0                   # right_hip_yaw
    cmd.motor_cmd[9].q = knee_target           # right_knee
    cmd.motor_cmd[10].q = ankle_pitch_target   # right_ankle_pitch
    cmd.motor_cmd[11].q = 0.0                  # right_ankle_roll

    print("Controlo ativo. A enviar trajetórias para os braços do G1...")
    t_start = time.time()

    try:
        while True:
            t = time.time() - t_start
            
            # Gerar onda sinusoidal (amplitude de ~25 graus a 0.5 Hz)
            sine_wave = 0.4 * math.sin(2 * math.pi * 0.5 * t)
            
            # De acordo com o IDL do G1, os motores de 12 a 22 cobrem a cinemática dos braços
            for i in range(12, 22):
                cmd.motor_cmd[i].q = sine_wave

            # Publicar no barramento de dados virtual
            low_cmd_puber.Write(cmd)
            time.sleep(0.005) # Loop a 200 Hz

    except KeyboardInterrupt:
        print("\nControlo encerrado.")