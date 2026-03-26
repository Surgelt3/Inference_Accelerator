#include <ARM_A9_HPS_arm_a9_0.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <thread>
#include <atomic>


void *virtual_base;

/*
 * Macros for device 'NPU_TOP_0_avs_reset', class 'NPU_TOP'
 * The macros are prefixed with 'NPU_TOP_0_AVS_RESET_'.
 * The prefix is the slave descriptor.
 */
#define NPU_TOP_0_AVS_RESET_COMPONENT_TYPE NPU_TOP
#define NPU_TOP_0_AVS_RESET_COMPONENT_NAME NPU_TOP_0
#define NPU_TOP_0_AVS_RESET_BASE 0xc0000000
#define NPU_TOP_0_AVS_RESET_SPAN 4
#define NPU_TOP_0_AVS_RESET_END 0xc0000003

/*
 * Macros for device 'NPU_TOP_0_avs_write', class 'NPU_TOP'
 * The macros are prefixed with 'NPU_TOP_0_AVS_WRITE_'.
 * The prefix is the slave descriptor.
 */
#define NPU_TOP_0_AVS_WRITE_COMPONENT_TYPE NPU_TOP
#define NPU_TOP_0_AVS_WRITE_COMPONENT_NAME NPU_TOP_0
#define NPU_TOP_0_AVS_WRITE_BASE 0xc0000004
#define NPU_TOP_0_AVS_WRITE_SPAN 4
#define NPU_TOP_0_AVS_WRITE_END 0xc0000007

/*
 * Macros for device 'NPU_TOP_0_avs_read', class 'NPU_TOP'
 * The macros are prefixed with 'NPU_TOP_0_AVS_READ_'.
 * The prefix is the slave descriptor.
 */
#define NPU_TOP_0_AVS_READ_COMPONENT_TYPE NPU_TOP
#define NPU_TOP_0_AVS_READ_COMPONENT_NAME NPU_TOP_0
#define NPU_TOP_0_AVS_READ_BASE 0xc0000008
#define NPU_TOP_0_AVS_READ_SPAN 8
#define NPU_TOP_0_AVS_READ_END 0xc000000f

/*
 * Macros for device 'NPU_TOP_0_avs_write_instr', class 'NPU_TOP'
 * The macros are prefixed with 'NPU_TOP_0_AVS_WRITE_INSTR_'.
 * The prefix is the slave descriptor.
 */
#define NPU_TOP_0_AVS_WRITE_INSTR_COMPONENT_TYPE NPU_TOP
#define NPU_TOP_0_AVS_WRITE_INSTR_COMPONENT_NAME NPU_TOP_0
#define NPU_TOP_0_AVS_WRITE_INSTR_BASE 0xc0000010
#define NPU_TOP_0_AVS_WRITE_INSTR_SPAN 4
#define NPU_TOP_0_AVS_WRITE_INSTR_END 0xc0000013


volatile uint32_t *reset_reg;
volatile uint32_t *write_reg;
volatile uint32_t *read_data;
volatile uint32_t *read_pc;
volatile uint32_t *writeinstr_reg;

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)

uint32_t instr_arr[] = {

};

typedef union {
    float f;
    uint32_t u;
} float_bits_t;



int main(void){

    uint32_t arr[] = {
        0x3AF8AE7D, 0xBA2300D4, 0x3B0BC921, 0x3B9A904F, 0xBB6F62CF, 0x3A815442, 0x3AC6F547,
        0xBBB018CC, 
		0xBB0AE11A, 
		0x3E000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3AB72546, 
		0xBA34ED40, 
		0x3AAE45F9, 
		0xBADDA48B, 
		0x3D061A61, 
		0x38CEED37, 
		0xBA928782, 
		0xBBB018CC, 
		0x3BEC51E9, 
		0xBF400000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E38B8AE, 
		0x3E38B8AE, 
		0x00000000, 
		0x3E048494, 
		0x3E008073, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000
		/* 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E38B8AE, 
		0x3E38B8AE, 
		0x3E3CBCCF, 
		0x3E048494, 
		0x3E008073, 
		0x3DF8F92B, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E38B8AE, 
		0x3E3CBCCF, 
		0x3E44C4CE, 
		0x3E008073, 
		0x3DF8F92B, 
		0x3DF8F92B, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E3CBCCF, 
		0x3E44C4CE, 
		0x3E40C0AD, 
		0x3DF8F92B, 
		0x3DF8F92B, 
		0x3DF0F0E9, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E44C4CE, 
		0x3E40C0AD, 
		0x3E3CBCCF, 
		0x3DF8F92B, 
		0x3DF0F0E9, 
		0x3DE8E8A7, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E40C0AD, 
		0x3E3CBCCF, 
		0x3E34B4D0, 
		0x3DF0F0E9, 
		0x3DE8E8A7, 
		0x3DD8D8A9, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E3CBCCF, 
		0x3E34B4D0, 
		0x3E30B0AF, 
		0x3DE8E8A7, 
		0x3DD8D8A9, 
		0x3DD0D0EE, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E34B4D0, 
		0x3E30B0AF, 
		0x3E3CBCCF, 
		0x3DD8D8A9, 
		0x3DD0D0EE, 
		0x3DB8B8BC, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E30B0AF, 
		0x3E3CBCCF, 
		0x3E38B8AE, 
		0x3DD0D0EE, 
		0x3DB8B8BC, 
		0x3DB0B0AF, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E3CBCCF, 
		0x3E38B8AE, 
		0x3E28A8B1, 
		0x3DB8B8BC, 
		0x3DB0B0AF, 
		0x3DA8A8A3, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E38B8AE, 
		0x3E28A8B1, 
		0x3E24A490, 
		0x3DB0B0AF, 
		0x3DA8A8A3, 
		0x3DB0B0AF, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E28A8B1, 
		0x3E24A490, 
		0x3E20A0B2, 
		0x3DA8A8A3, 
		0x3DB0B0AF, 
		0x3DB0B0AF, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E24A490, 
		0x3E20A0B2, 
		0x3E149492, 
		0x3DB0B0AF, 
		0x3DB0B0AF, 
		0x3DA8A8A3, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E20A0B2, 
		0x3E149492, 
		0x3E109071, 
		0x3DB0B0AF, 
		0x3DA8A8A3, 
		0x3D989898, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E149492, 
		0x3E109071, 
		0x3E109071, 
		0x3DA8A8A3, 
		0x3D989898, 
		0x3DA0A0A4, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x3E109071, 
		0x3E109071, 
		0x3E20A0B2, 
		0x3D989898, 
		0x3DA0A0A4, 
		0x3D88888D, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000, 
		0x00000000*/
    };

    //*(volatile unsigned int *)NPU_TOP_0_AVS_RESET_BASE = 0x1;
    //*(volatile unsigned int *)NPU_TOP_0_AVS_RESET_BASE = 0x0;
    int fd;
    void *virtual_base;

    //open the /dev/mem to access the FPGA space for reading and writing
    if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) {
        printf( "ERROR: could not open \"/dev/mem\"...\n" );
        return( 1 );
    }
    //map the virtual memory space to virtual_base, that is 2MB in size
    //(0x00200000), at address LWHPS2FPGA_BASE
    virtual_base = mmap( NULL, MAP_SIZE, ( PROT_READ | PROT_WRITE ),
    MAP_SHARED, fd, NPU_TOP_0_AVS_RESET_BASE & ~MAP_MASK);

    if (virtual_base == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // map the address space for the LED and HEX registers into user space so
    //we can interact with them.. virtual_base + the offset of your IP component
    reset_reg = (volatile uint32_t *)((char*)virtual_base + (NPU_TOP_0_AVS_RESET_BASE & MAP_MASK));
    write_reg = (volatile uint32_t *)((char*)virtual_base + (NPU_TOP_0_AVS_WRITE_BASE & MAP_MASK));
    read_data  = (volatile uint32_t *)((char*)virtual_base + (NPU_TOP_0_AVS_READ_BASE  & MAP_MASK));
    read_pc = read_data + 1;
	writeinstr_reg = (volatile uint32_t *)((char*)virtual_base + (NPU_TOP_0_AVS_WRITE_INSTR_BASE  & MAP_MASK));



    unsigned int prev = 0;
    uint32_t top_u32, bot_u32;
    float_bits_t conv;

    float data;
    unsigned int pc;

    *reset_reg = 0x1;
    *reset_reg = 0x0;

	std::atomic<bool> data_writer_done{false};
	std::atomic<bool> instr_writer_done{false};


    auto writer = [&]() {
		size_t num_writes = sizeof(arr) / sizeof(arr[0]);
        for (int i = 0; i < num_writes; i++) {
            *write_reg = arr[i];
        }
        data_writer_done.store(true);

    };

    auto reader = [&]() {
        uint32_t top_u32, bot_u32;
        float_bits_t conv;
        float data;
        unsigned int pc;
        int counter = 0;

        while (1) {
            bot_u32 = *read_data;
            top_u32 = *read_pc;

            pc = top_u32;
            conv.u = bot_u32;
            data = conv.f;

            printf("iter: %d %u: %f\n", counter, pc, data);

            counter++;
            if (pc == 8) {
                break;
            }

            // Optional: small delay to avoid hammering the bus
            // usleep(100);
        }
    };

	auto write_instr = [&]() {
		size_t instr_len = sizeof(instr_arr) / sizeof(instr_arr[0]);

		for (int i = 0; i < instr_len; i++){
			*writeinstr_reg = instr_arr[i];
		}
        instr_writer_done.store(true);

    };

    std::thread writerinstr_thread(write_instr);
    std::thread writer_thread(writer);
    std::thread reader_thread(reader);

	writerinstr_thread.join();
    writer_thread.join();
    reader_thread.join();

    munmap(virtual_base, MAP_SIZE);
    close(fd);
    return 0;	



}