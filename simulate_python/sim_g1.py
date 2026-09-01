import time
import mujoco
import mujoco.viewer
from threading import Thread
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py_bridge import ElasticBand  # Importar a classe original da Unitree
import config

# 1. Carregar o modelo e o cenário completo
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = config.SIMULATE_DT

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    elastic_band.enable = True  # Forçar ativo por defeito já que não há key_callback
    
    # Identificar o link correto para aplicar a força de suspensão
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id

# 2. Variável partilhada para reter o último comando recebido via DDS
ultimo_comando = None

def LowCmdHandler(msg: LowCmd_):
    global ultimo_comando
    ultimo_comando = msg

def dds_listener_thread():
    """Thread dedicada exclusivamente a processar e escutar a rede DDS."""
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(LowCmdHandler, 10)
    
    while True:
        time.sleep(0.1)

# 3. Callback global do MuJoCo (Onde fundimos o Controlo Motor + ElasticBand)
def my_control_callback(model, data):
    global ultimo_comando
    
    # APLICAR FORÇA ELASTIC BAND
    if config.ENABLE_ELASTIC_BAND:
        if elastic_band.enable:
            data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                data.qpos[:3], data.qvel[:3]
            )

    # --- APLICAR COMANDOS MOTOR DDS ---
    if ultimo_comando is not None:
        for i in range(min(model.nu, 35)):
            if ultimo_comando.motor_cmd[i].mode == 10:    # Controlo de posição
                data.ctrl[i] = ultimo_comando.motor_cmd[i].q
            elif ultimo_comando.motor_cmd[i].mode == 1:    # Controlo de binário
                data.ctrl[i] = ultimo_comando.motor_cmd[i].tau

if __name__ == "__main__":
    print("A iniciar a thread de escuta DDS in background...")
    network_thread = Thread(target=dds_listener_thread)
    network_thread.daemon = True
    network_thread.start()

    # 4. Registar o callback unificado diretamente no motor do MuJoCo
    mujoco.set_mjcb_control(my_control_callback)
    
    # 5. Lançar o visualizador bloqueante estável com a assinatura limpa da API
    mujoco.viewer.launch(mj_model, mj_data)