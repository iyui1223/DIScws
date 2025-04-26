/**
 * @file heat_diffusion.cpp
 * @brief 2D Heat Diffusion Simulation using the Finite Difference Method
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <chrono>
 #include <thread>
 #include <fstream>
 #include <string>
 
 class HeatDiffusion {
 private:
     int width, height;
     double diffusionRate;
 
     std::vector<std::vector<double>> temperature;
     std::vector<std::vector<double>> nextTemperature;
 
     void saveFrame(int frameNumber) {
         system("mkdir -p output");
         auto start = std::chrono::high_resolution_clock::now();
         
         std::string filename = "output/frame_" + std::to_string(frameNumber) + ".txt";
         std::ofstream outFile(filename);
         
         if (outFile.is_open()) {
             for (const auto& row : temperature) {
                 for (const auto& temp : row) {
                     outFile << temp << " ";
                 }
                 outFile << "\n"; // Ensure new lines are added
             }
             outFile.close();
         }
         
         auto end = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = end - start;
         std::cout << "Text output time: " << elapsed.count() << " seconds" << std::endl;
     }
 
 public:
     HeatDiffusion(int w, int h, double rate) : width(w), height(h), diffusionRate(rate) {
         auto start_init = std::chrono::high_resolution_clock::now();
 
         // Initialize grids with ambient temperature (20°C)
         auto start_vector_alloc = std::chrono::high_resolution_clock::now();
         temperature = std::vector<std::vector<double>>(height, std::vector<double>(width, 20.0));
         nextTemperature = temperature;
         auto end_vector_alloc = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> vector_alloc_time = end_vector_alloc - start_vector_alloc;
         std::cout << "Vector allocation time: " << vector_alloc_time.count() << " seconds" << std::endl;
 
         // Set up initial conditions - hot spot in the middle (100°C)
         auto start_hotspot = std::chrono::high_resolution_clock::now();
         int centerX = width / 2;
         int centerY = height / 2;
         for (int y = centerY - 3; y <= centerY + 3; y++) {
             for (int x = centerX - 3; x <= centerX + 3; x++) {
                 if (y >= 0 && y < height && x >= 0 && x < width) {
                     temperature[y][x] = 100.0;
                 }
             }
         }
         auto end_hotspot = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> hotspot_time = end_hotspot - start_hotspot;
         std::cout << "Hotspot initialization time: " << hotspot_time.count() << " seconds" << std::endl;
 
         auto end_init = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> init_time = end_init - start_init;
         std::cout << "Total initialization time: " << init_time.count() << " seconds" << std::endl;
     }
 
     void update() {
         static int frameCount = 0;
         auto start = std::chrono::high_resolution_clock::now();          
  
         for (int y = 1; y < height - 1; y++) {
             for (int x = 1; x < width - 1; x++) {
                 double laplacian = 
                     temperature[y+1][x] + temperature[y-1][x] + 
                     temperature[y][x+1] + temperature[y][x-1] - 
                     4 * temperature[y][x];
 
                 nextTemperature[y][x] = temperature[y][x] + diffusionRate * laplacian;
             }
         }
 
         // Swap buffers
         temperature.swap(nextTemperature);
         auto end = std::chrono::high_resolution_clock::now();
         
         std::chrono::duration<double> elapsed = end - start;

 
         // Save frame every 10 timesteps
         if (frameCount % 10 == 0) {
             std::cout << "Diffusion loop time: " << elapsed.count() << " seconds" << std::endl;
             saveFrame(frameCount / 10);
         }
         frameCount++;
     }
 };
 
 int main(int argc, char* argv[]) {
     auto start_time = std::chrono::high_resolution_clock::now();
 
     std::string exp_name = std::getenv("EXP_NAME") ? std::getenv("EXP_NAME") : "unknown";
     int width = std::stoi(argv[1]);
     int height = std::stoi(argv[2]);
     const int total_frames = std::stoi(argv[3]);
 
     HeatDiffusion simulation(width, height, 0.1);
 
     for (int i = 0; i < total_frames; i++) {
         simulation.update();
     }
 
     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> elapsed_time = end_time - start_time;
     std::cout << "Total execution time: " << elapsed_time.count() << " seconds\n";
 
     std::ofstream logFile("experiment_log.csv", std::ios::app);
     logFile << exp_name << "," << width << "," << height << "," 
             << total_frames << "," << elapsed_time.count() << "\n";
     logFile.close();
 
     return 0;
 }
 