from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Camera, Electrics
import numpy as np
import cv2


class SimulationManager:
    def __init__(
        self,
        beamng_home=r"C:\Users\Benjamin\Applications\BeamNG.tech.v0.37.6.0",
        host="localhost",
        port=25252,
    ):
        self.bng = BeamNGpy(host, port, home=beamng_home)
        self.scenario = None
        self.vehicle = None
        self.camera = None

    def setup_scenario(self, scenario_name="lane_detection_test"):
        self.bng.open()

        self.scenario = Scenario("west_coast_usa", scenario_name)

        self.vehicle = Vehicle("ego", model="etk800", license="CS566")

        self.scenario.add_vehicle(
            self.vehicle, pos=(-823, 2, 118), rot_quat=(0, 0, 0.38, 0.92)
        )

        self.scenario.make(self.bng)
        self.bng.settings.set_deterministic(60)
        self.bng.control.pause()
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()
        self.bng.control.resume()

        cam_res = (1280, 720)
        self.camera = Camera(
            "front_cam",
            self.bng,
            self.vehicle,
            pos=(0.0, -2, 1),
            dir=(0, 0, 0),
            field_of_view_y=70,
            resolution=cam_res,
            requested_update_time=0.05,
            near_far_planes=(0.1, 1000),
            is_visualised=True,
            is_using_shared_memory=True,
            is_render_annotations=True,
            is_streaming=True,
            is_render_colours=True,
        )
        self.electrics = Electrics()
        self.vehicle.attach_sensor("electrics", self.electrics)

        self.vehicle.ai.set_mode("traffic")

    def get_frame(self):
        """Returns the latest camera frame (BGR) and steering angle."""
        if not self.camera:
            print("Camera not initialized.")
            return None, 0.0

        try:
            readings = self.camera.stream()
            sensors = self.vehicle.sensors
            sensors.poll()

            image = np.array(readings["colour"])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            steering = sensors["electrics"]["steering_input"]

            return image, steering
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, 0.0

    def pause(self):
        self.bng.control.pause()

    def resume(self):
        self.bng.control.resume()

    def close(self):
        self.bng.close()
