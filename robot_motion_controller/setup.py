from setuptools import find_packages, setup

package_name = 'robot_motion_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parakram',
    maintainer_email='parakram@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_follower = robot_motion_controller.person_follower:main',
            'graph_path = robot_motion_controller.graph_path:main',
        ],
    },
)
