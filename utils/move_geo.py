import airsim
import time

# --- Target NED Coordinates (in meters) ---
TARGET_X = 595.63  # North
TARGET_Y = -258.19 # East (Negative is West)
TARGET_Z = -0.68   # Down (Negative is Up/Above ground plane)

# Use the empty string as vehicle name for compatibility with older AirSim servers
VEHICLE_NAME = "" 
# -----------------------------------------

if __name__ == "__main__":
    try:
        client = airsim.CarClient()
        client.confirmConnection()
        print("Connected to AirSim!")

        # 1. Create a Vector3r object for the target position
        target_position = airsim.Vector3r(TARGET_X, TARGET_Y, TARGET_Z)
        
        # 2. Get the current orientation to maintain the car's direction
        # This prevents the car from suddenly facing a new direction upon teleport.
        current_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
        current_orientation = current_pose.orientation

        # 3. Create the target pose (position + current orientation)
        target_pose = airsim.Pose(target_position, current_orientation)
        
        # 4. Set the vehicle's pose
        # ignore_collision=True allows teleportation without the physics engine flagging a collision.
        client.simSetVehiclePose(target_pose, ignore_collision=True, vehicle_name=VEHICLE_NAME)
        
        print("\n✅ Car successfully teleported.")
        print(f"   NED Position Set To: X={TARGET_X:.2f}m, Y={TARGET_Y:.2f}m, Z={TARGET_Z:.2f}m")

        # Optional: Verify the new position after a short delay
        time.sleep(0.5)
        new_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
        
        # Print actual position for verification
        print(f"\nVerification:")
        print(f"   Current X (North): {new_pose.position.x_val:.2f}m")
        print(f"   Current Y (East): {new_pose.position.y_val:.2f}m")
        print(f"   Current Z (Down): {new_pose.position.z_val:.2f}m")

    except Exception as e:
        print(f"\n❌ Error setting vehicle pose. Ensure the simulator is running.")
        print(f"   Details: {e}")