import airsim
import time

# --- Configuration ---
# Use the empty string as vehicle name for compatibility with older AirSim servers
VEHICLE_NAME = "" 
# ---------------------

if __name__ == "__main__":
    try:
        client = airsim.CarClient() 
        client.confirmConnection()
        print("Connected to AirSim!")
        
    except Exception as e:
        print(f"Error connecting to AirSim. Ensure the simulator is running in Car mode: {e}")
        exit()

    last_print_time = time.time()
    
    print("\n--- NED Position Tracker Active ---")
    print("Coordinates are in meters relative to the car's starting point.")
    print("Press Ctrl+C to quit.")

    while True:
        try:
            current_time = time.time()
            
            # Check if 2 seconds have passed since the last print
            if current_time - last_print_time >= 2.0:
                
                # Retrieve the vehicle's current pose (position and orientation)
                # Using the old API fix with vehicle_name=""
                pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
                
                # Extract the position vector
                position = pose.position
                
                # NED Coordinates (in meters)
                ned_x = position.x_val
                ned_y = position.y_val
                ned_z = position.z_val
                
                # Print the data
                print(f"Time: {time.strftime('%H:%M:%S', time.localtime())} | X (North): {ned_x:.2f}m | Y (East): {ned_y:.2f}m | Z (Down): {ned_z:.2f}m")
                    
                last_print_time = current_time
            
            # Use a short sleep to prevent the loop from consuming too much CPU
            time.sleep(0.1) 
            
        except KeyboardInterrupt:
            print("\nExiting tracker.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the loop: {e}")
            break