/**
 * @file heat_diffusion.cpp
 * @brief 2D Heat Diffusion Simulation using the Finite Difference Method
 */

 #include <iostream>
 #include <vector>
 #include <fstream>
 #include <cstdlib> // for std::getenv()
 #include <chrono>
 #include <cassert>
 
 /**
  * @brief 2D array class for handling 1d vector as 2d.
  */
 template <typename T>
 class Array2D {
 private:
     int size1;
     int size2;
     T* data;
 
 public:
    Array2D(int size1, int size2, T init_val = T()) {
        this->size1 = size1;
        this->size2 = size2;
        this->data = new T[size1 * size2];

        for (int i = 0; i < size1 * size2; i++) {
            this->data[i] = init_val;
        }
    }
 
     ~Array2D() {
         delete[] this->data;
     }
 
     T& operator()(int i, int j) {
         return this->data[i * size2 + j];
     }
 
     T* getPointer() {
         return this->data;
     }
 
     int getSize() {
         return this->size1 * this->size2;
     }
     void swap(Array2D<T>& other) {
        std::swap(this->data, other.data);
    }     
 };
 
 /**
  * @class HeatDiffusion
  * @brief Class implementing the 2D heat diffusion simulation.
  */
 class HeatDiffusion {
 private:
     int width, height;
     float diffusionRate;
     Array2D<float> temperature;
     Array2D<float> nextTemperature;
     
     /**
      * @brief Saves the current temperature grid to a binary file.
      */
     void saveFrame(int frameNumber) {
         system("mkdir -p output");
         auto start = std::chrono::high_resolution_clock::now();
         std::string filename = "output/frame_" + std::to_string(frameNumber) + ".bin";
         std::ofstream outFile(filename, std::ios::binary);
         outFile.write(reinterpret_cast<const char*>(temperature.getPointer()), temperature.getSize() * sizeof(float));
         outFile.close();
         auto end = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> elapsed = end - start;
         std::cout << "Binary output: " << elapsed.count() << " seconds" << std::endl;
     }
 
 public:
     /**
      * @brief Constructor initializes a heat diffusion grid.
      */
     HeatDiffusion(int w, int h, float rate)
         : width(w), height(h), diffusionRate(rate),
           temperature(h, w, 20.0), 
           nextTemperature(h, w, 20.0) {
          
         // Initial conditions
         int centerX = width / 2;
         int centerY = height / 2;
         for (int y = centerY - 3; y <= centerY + 3; y++) {
             for (int x = centerX - 3; x <= centerX + 3; x++) {
                 if (y >= 0 && y < height && x >= 0 && x < width) {
                     temperature(y, x) = 100.0;
                 }
             }
         }
     }
 
     /**
      * @brief Updates the simulation by one timestep.
      */
     void update() {
         static int frameCount = 0;
         auto start = std::chrono::high_resolution_clock::now(); 
         for (int y = 1; y < height - 1; y++) {
             for (int x = 1; x < width - 1; x++) {
                 float laplacian =
                     temperature(y + 1, x) + temperature(y - 1, x) +
                     temperature(y, x + 1) + temperature(y, x - 1) -
                     4.0 * temperature(y, x);
 
                 nextTemperature(y, x) = temperature(y, x) + diffusionRate * laplacian;
             }
         }
 
         temperature.swap(nextTemperature);
         auto end = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> elapsed = end - start;

         // Save every 10 frames
         if (frameCount % 10 == 0) {
            std::cout << "diffution loop: " << elapsed.count() << " seconds" << std::endl;
            saveFrame(frameCount / 10);
         }
         frameCount++;
     }
 };
 
 /**
  * @brief Main function running the heat diffusion simulation.
  */
 int main(int argc, char* argv[]) {
     auto start_time = std::chrono::high_resolution_clock::now();
 
     // Read experiment name from environment variable
     std::string exp_name = std::getenv("EXP_NAME") ? std::getenv("EXP_NAME") : "unknown";
 
     if (argc != 4) {
         std::cerr << "Usage: " << argv[0] << " <width> <height> <frames>\n";
         return 1;
     }
 
     int width = std::stoi(argv[1]);
     int height = std::stoi(argv[2]);
     int total_frames = std::stoi(argv[3]);
 
     HeatDiffusion simulation(width, height, 0.1);
 
     for (int i = 0; i < total_frames; i++) {
         simulation.update();
     }
 
     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<float> elapsed_time = end_time - start_time;
     std::cout << "Total execution time: " << elapsed_time.count() << " seconds\n";

     // Append results to CSV log
     std::ofstream logFile("experiment_log.csv", std::ios::app);
     logFile << exp_name << "," << width << "," << height << ","
             << total_frames << "," << elapsed_time.count() << "\n";
     logFile.close();
 
     return 0;
 }
 