import openttd
from openttdlab import run_experiments, local_file

experiments = [
    {
        'seed': 12345,  # 随机种子
        'days': 365,  # 运行天数
        'ais': [
            ('my_ai', {'param1': 'value1'}, local_file('path/to/ai.tar', 'my_ai')),
        ],
        'openttd_config': '''
            [network]
            server_name = My Server
        '''
    }
]

results = run_experiments(
    experiments=experiments,
    final_screenshot_directory='path/to/screenshots',
    openttd_version='12.0',  # 指定 OpenTTD 版本
    opengfx_version='7.0'  # 指定 OpenGFX 版本
)

for result in results:
    print(result)
