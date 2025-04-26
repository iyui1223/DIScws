/**
 * @file heat_diffusion.cpp
 * @brief 2D Heat Diffusion Simulation using the Finite Difference Method
 */
#include <mpi.h>
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
    Array2D<float> local_temp;
    Array2D<float> local_nextTemp;

    int my_rank, num_procs;
    int rows_per_proc;
    int buff;

    void saveFrame(int frameNumber) {
        system("mkdir -p output");
        auto start = std::chrono::high_resolution_clock::now();
        std::string filename = "output/frame_" + std::to_string(frameNumber) + ".bin";
        std::ofstream outFile(filename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(temperature.getPointer()), temperature.getSize() * sizeof(float));
        outFile.close();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        // std::cout << "Binary output: " << elapsed.count() << " seconds" << std::endl;
    }

public:
    HeatDiffusion(int w, int h, float rate, int my_rank, int num_procs)
        : width(w), height(h), diffusionRate(rate),
          temperature(h, w, 20.0),
          nextTemperature(h, w, 20.0),
          local_temp((h / num_procs) + 2, w, 0.0),
          local_nextTemp((h / num_procs) + 2, w, 0.0),
          my_rank(my_rank), num_procs(num_procs) {

        rows_per_proc = height / num_procs;
        buff = rows_per_proc * width;
        // local_temp = Array2D<float>(rows_per_proc + 2, width, 0.0);
        // local_nextTemp = Array2D<float>(rows_per_proc + 2, width, 0.0);

        if (my_rank == 0) {
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

        MPI_Bcast(temperature.getPointer(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

        MPI_Scatter(
            temperature.getPointer(), rows_per_proc * width, MPI_FLOAT,
            local_temp.getPointer() + width, rows_per_proc * width, MPI_FLOAT,
            0, MPI_COMM_WORLD);
    }

    void update() {
        static int frameCount = 0;
        auto start = std::chrono::high_resolution_clock::now();

        if (my_rank < num_procs - 1) {
            MPI_Sendrecv(
                local_temp.getPointer() + width * rows_per_proc,
                width, MPI_FLOAT, my_rank + 1, 0,
                local_temp.getPointer() + width * (rows_per_proc + 1),
                width, MPI_FLOAT, my_rank + 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (my_rank > 0) {
            MPI_Sendrecv(
                local_temp.getPointer() + width,
                width, MPI_FLOAT, my_rank - 1, 1,
                local_temp.getPointer(),
                width, MPI_FLOAT, my_rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int y = 1; y <= rows_per_proc; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float laplacian =
                    local_temp(y + 1, x) + local_temp(y - 1, x) +
                    local_temp(y, x + 1) + local_temp(y, x - 1) -
                    4.0f * local_temp(y, x);

                local_nextTemp(y, x) = local_temp(y, x) + diffusionRate * laplacian;
            }
        }

        local_temp.swap(local_nextTemp);

        MPI_Gather(
            local_temp.getPointer() + width,
            rows_per_proc * width, MPI_FLOAT,
            temperature.getPointer(),
            rows_per_proc * width, MPI_FLOAT,
            0, MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;

        if (my_rank == 0 && frameCount % 10 == 0) {
            // std::cout << "Diffusion loop (frame " << frameCount << "): "
            //          << elapsed.count() << " seconds" << std::endl;
            saveFrame(frameCount / 10);
        }

        frameCount++;
    }
};

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string exp_name = std::getenv("EXP_NAME") ? std::getenv("EXP_NAME") : "unknown";

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <frames>\n";
        return 1;
    }

    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int total_frames = std::stoi(argv[3]);

    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HeatDiffusion simulation(width, height, 0.1f, my_rank, num_procs);

    for (int i = 0; i < total_frames; ++i) {
        simulation.update();
    }

    MPI_Finalize();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_time = end_time - start_time;
    std::cout << "Total execution time: " << elapsed_time.count() << " seconds\n";

    std::ofstream logFile("experiment_log.csv", std::ios::app);
    logFile << exp_name << "," << width << "," << height << ","
            << total_frames << "," << elapsed_time.count() << "\n";
    logFile.close();

    return 0;
}
