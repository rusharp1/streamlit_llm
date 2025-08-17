#define ZIG_TARGET_MAX_INT_ALIGNMENT 16
#include "zig.h"
struct anon__lazy_57 {
 uint8_t const *ptr;
 uintptr_t len;
};
struct fifo_UnixPipe_Writer__538;
typedef struct anon__lazy_65 nav__987_39;
struct mem_Allocator__206;
typedef struct anon__lazy_57 nav__987_42;
struct mem_Allocator_VTable__209;
struct mem_Allocator__206 {
 void *ptr;
 struct mem_Allocator_VTable__209 const *vtable;
};
struct fs_File__546;
struct fs_File__546 {
 int32_t handle;
};
struct fifo_UnixPipe_Writer__538 {
 struct mem_Allocator__206 allocator;
 struct fs_File__546 file;
};
struct anon__lazy_65 {
 struct fifo_UnixPipe_Writer__538 payload;
 uint16_t error;
};
struct fs_File_OpenFlags__1807;
struct fs_File_OpenFlags__1807 {
 uint8_t mode;
 uint8_t lock;
 bool lock_nonblocking;
 bool allow_ctty;
};
typedef struct anon__lazy_79 nav__987_56;
struct anon__lazy_79 {
 struct fs_File__546 payload;
 uint16_t error;
};
struct fifo_UnixPipe_Reader__540;
typedef struct anon__lazy_82 nav__986_39;
typedef struct anon__lazy_57 nav__986_42;
struct fifo_UnixPipe_Reader__540 {
 struct mem_Allocator__206 allocator;
 struct fs_File__546 file;
};
struct anon__lazy_82 {
 struct fifo_UnixPipe_Reader__540 payload;
 uint16_t error;
};
typedef struct anon__lazy_79 nav__986_56;
struct instruments_perf_PerfInstrument__200;
typedef struct anon__lazy_88 nav__337_39;
struct instruments_perf_PerfInstrument__200 {
 struct mem_Allocator__206 allocator;
 struct fifo_UnixPipe_Writer__538 writer;
 struct fifo_UnixPipe_Reader__540 reader;
};
struct anon__lazy_88 {
 struct instruments_perf_PerfInstrument__200 payload;
 uint16_t error;
};
typedef struct anon__lazy_57 nav__337_59;
typedef struct anon__lazy_65 nav__337_61;
typedef struct anon__lazy_82 nav__337_66;
struct instruments_root_InstrumentHooks__188;
typedef struct anon__lazy_99 nav__314_39;
struct instruments_valgrind_ValgrindInstrument__195;
struct instruments_valgrind_ValgrindInstrument__195 {
 struct mem_Allocator__206 allocator;
};
struct instruments_root_InstrumentHooks__188 {
 union {
  struct instruments_valgrind_ValgrindInstrument__195 valgrind;
  struct instruments_perf_PerfInstrument__200 perf;
 } payload;
 uint8_t tag;
};
struct anon__lazy_99 {
 struct instruments_root_InstrumentHooks__188 payload;
 uint16_t error;
};
typedef struct anon__lazy_88 nav__314_60;
typedef struct anon__lazy_109 nav__231_46;
struct anon__lazy_109 {
 struct instruments_root_InstrumentHooks__188 *payload;
 uint16_t error;
};
typedef struct anon__lazy_114 nav__231_51;
struct mem_Allocator_VTable__209 {
 uint8_t *(*alloc)(void *, uintptr_t, uint8_t, uintptr_t);
 bool (*resize)(void *, struct anon__lazy_114, uint8_t, uintptr_t, uintptr_t);
 uint8_t *(*remap)(void *, struct anon__lazy_114, uint8_t, uintptr_t, uintptr_t);
 void (*free)(void *, struct anon__lazy_114, uint8_t, uintptr_t);
};
typedef struct anon__lazy_99 nav__231_71;
typedef struct anon__lazy_109 nav__3488_40;
typedef struct anon__lazy_125 nav__3488_51;
struct anon__lazy_125 {
 uint8_t *payload;
 uint16_t error;
};
typedef struct anon__lazy_57 nav__1053_39;
struct fs_Dir__1809;
struct fs_Dir__1809 {
 int32_t fd;
};
typedef struct anon__lazy_79 nav__1050_39;
typedef struct anon__lazy_57 nav__1050_41;
struct shared_Command__1919;
struct shared_Command__struct_1922__1922;
typedef struct anon__lazy_57 nav__991_44;
struct shared_Command__struct_1922__1922 {
 struct anon__lazy_57 uri;
 uint32_t pid;
};
struct shared_Command__struct_1923__1923;
struct shared_Command__struct_1923__1923 {
 struct anon__lazy_57 name;
 struct anon__lazy_57 version;
};
struct shared_Command__1919 {
 union {
  struct shared_Command__struct_1922__1922 ExecutedBenchmark;
  struct shared_Command__struct_1923__1923 SetIntegration;
 } payload;
 uint8_t tag;
};
struct array_list_ArrayListAligned_28u8_2cnull_29__1950;
typedef struct anon__lazy_114 nav__991_55;
struct anon__lazy_114 {
 uint8_t *ptr;
 uintptr_t len;
};
struct array_list_ArrayListAligned_28u8_2cnull_29__1950 {
 struct anon__lazy_114 items;
 uintptr_t capacity;
 struct mem_Allocator__206 allocator;
};
struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981;
struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *context;
};
typedef struct anon__lazy_159 nav__4169_38;
struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156;
struct anon__lazy_159 {
 uint32_t payload;
 uint16_t error;
};
struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133;
struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *context;
};
struct io_Reader__2282;
typedef struct anon__lazy_171 nav__4169_50;
typedef struct anon__lazy_114 nav__4169_52;
struct io_Reader__2282 {
 void const *context;
 struct anon__lazy_171 (*readFn)(void const *, struct anon__lazy_114);
};
struct anon__lazy_171 {
 uintptr_t payload;
 uint16_t error;
};
typedef struct anon__lazy_181 nav__4169_64;
struct anon__lazy_181 {
 uint16_t error;
 uint8_t payload[4];
};
typedef struct anon__lazy_159 nav__4168_38;
typedef struct anon__lazy_114 nav__4211_39;
typedef struct anon__lazy_185 nav__4211_40;
struct anon__lazy_185 {
 struct anon__lazy_114 payload;
 uint16_t error;
};
typedef struct anon__lazy_171 nav__4211_59;
typedef struct anon__lazy_189 nav__4211_69;
struct anon__lazy_189 {
 uint64_t payload;
 uint16_t error;
};
typedef struct anon__lazy_192 nav__4211_72;
struct anon__lazy_192 {
 uint16_t error;
 uint8_t payload[8];
};
typedef struct anon__lazy_57 nav__4210_39;
typedef struct anon__lazy_195 nav__4210_40;
struct anon__lazy_195 {
 struct anon__lazy_57 payload;
 uint16_t error;
};
typedef struct anon__lazy_114 nav__4210_55;
typedef struct anon__lazy_185 nav__4210_57;
typedef struct anon__lazy_199 nav__4209_39;
typedef struct anon__lazy_57 nav__4209_44;
struct anon__lazy_199 {
 struct shared_Command__struct_1922__1922 payload;
 uint16_t error;
};
typedef struct anon__lazy_159 nav__4209_57;
typedef struct anon__lazy_195 nav__4209_60;
typedef struct anon__lazy_199 nav__4208_39;
typedef struct anon__lazy_57 nav__4208_44;
typedef struct anon__lazy_207 nav__4216_39;
typedef struct anon__lazy_57 nav__4216_44;
struct anon__lazy_207 {
 struct shared_Command__struct_1923__1923 payload;
 uint16_t error;
};
typedef struct anon__lazy_195 nav__4216_57;
typedef struct anon__lazy_207 nav__4215_39;
typedef struct anon__lazy_57 nav__4215_44;
typedef struct anon__lazy_211 nav__4167_39;
typedef struct anon__lazy_57 nav__4167_45;
struct anon__lazy_211 {
 struct shared_Command__1919 payload;
 uint16_t error;
};
typedef struct anon__lazy_159 nav__4167_61;
typedef struct anon__lazy_199 nav__4167_63;
typedef struct anon__lazy_207 nav__4167_65;
typedef struct anon__lazy_211 nav__4166_39;
typedef struct anon__lazy_57 nav__4166_45;
typedef struct anon__lazy_211 nav__996_39;
typedef struct anon__lazy_57 nav__996_45;
typedef struct anon__lazy_114 nav__996_66;
typedef struct anon__lazy_171 nav__996_69;
typedef struct anon__lazy_185 nav__996_73;
struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 {
 struct anon__lazy_114 buffer;
 uintptr_t pos;
};
typedef struct anon__lazy_211 nav__997_39;
typedef struct anon__lazy_220 nav__997_42;
typedef struct anon__lazy_57 nav__997_46;
struct anon__lazy_220 {
 uint64_t payload;
 bool is_null;
};
typedef struct anon__lazy_220 nav__998_40;
typedef struct anon__lazy_57 nav__998_48;
typedef struct anon__lazy_211 nav__998_55;
typedef struct anon__lazy_57 nav__339_44;
typedef struct anon__lazy_220 nav__339_67;
typedef struct anon__lazy_57 nav__340_45;
typedef struct anon__lazy_125 nav__3737_39;
typedef struct anon__lazy_114 nav__3737_52;
typedef struct anon__lazy_57 nav__3772_39;
typedef struct anon__lazy_57 nav__3693_40;
typedef struct anon__lazy_244 nav__3693_49;
struct anon__lazy_244 {
 uint16_t error;
 uint8_t payload[4096];
};
typedef struct anon__lazy_79 nav__3629_39;
typedef struct anon__lazy_57 nav__3629_42;
typedef struct anon__lazy_244 nav__3629_53;
typedef struct anon__lazy_114 nav__3885_42;
typedef struct anon__lazy_57 nav__4058_46;
typedef struct anon__lazy_114 nav__3887_41;
typedef struct anon__lazy_57 nav__1144_40;
typedef struct anon__lazy_171 nav__1144_47;
struct os_linux_timespec__struct_2728__2728;
struct os_linux_timespec__struct_2728__2728 {
 intptr_t sec;
 intptr_t nsec;
};
typedef struct anon__lazy_267 nav__4072_41;
struct anon__lazy_267 {
 struct os_linux_timespec__struct_2728__2728 payload;
 uint16_t error;
};
typedef struct anon__lazy_171 nav__1134_38;
typedef struct anon__lazy_114 nav__1134_41;
typedef struct anon__lazy_114 nav__4100_39;
typedef struct anon__lazy_185 nav__4100_40;
typedef struct anon__lazy_125 nav__4100_51;
typedef struct anon__lazy_114 nav__4132_40;
typedef struct anon__lazy_171 nav__4165_38;
typedef struct anon__lazy_114 nav__4165_41;
typedef struct anon__lazy_181 nav__4207_39;
typedef struct anon__lazy_171 nav__4207_44;
typedef struct anon__lazy_114 nav__4207_46;
typedef struct anon__lazy_192 nav__4212_39;
typedef struct anon__lazy_171 nav__4212_44;
typedef struct anon__lazy_114 nav__4212_46;
typedef struct anon__lazy_171 nav__4172_38;
typedef struct anon__lazy_114 nav__4172_41;
typedef struct anon__lazy_57 nav__4213_40;
typedef struct anon__lazy_289 nav__4213_42;
struct anon__lazy_289 {
 uintptr_t payload;
 bool is_null;
};
typedef struct anon__lazy_114 nav__4217_40;
struct cimport_struct_timespec__2902;
struct cimport_struct_timespec__2902 {
 long tv_sec;
 long tv_nsec;
};
typedef struct anon__lazy_299 nav__4222_42;
struct anon__lazy_299 {
 long payload;
 bool is_null;
};
typedef struct anon__lazy_57 nav__3835_43;
typedef struct anon__lazy_57 nav__3778_39;
typedef struct anon__lazy_244 nav__1628_39;
typedef struct anon__lazy_57 nav__1628_41;
typedef struct anon__lazy_114 nav__1628_47;
typedef struct anon__lazy_79 nav__3630_39;
typedef struct anon__lazy_312 nav__3630_50;
struct anon__lazy_312 {
 int32_t payload;
 uint16_t error;
};
typedef struct anon__lazy_57 nav__4236_46;
typedef struct anon__lazy_114 nav__3933_39;
typedef struct anon__lazy_171 nav__1143_38;
typedef struct anon__lazy_57 nav__1143_41;
typedef struct anon__lazy_267 nav__1544_39;
typedef struct anon__lazy_171 nav__1133_38;
typedef struct anon__lazy_114 nav__1133_41;
typedef struct anon__lazy_125 nav__4237_39;
typedef struct anon__lazy_171 nav__4237_49;
typedef struct anon__lazy_171 nav__4124_38;
typedef struct anon__lazy_114 nav__4124_42;
typedef struct anon__lazy_114 nav__4174_40;
typedef struct anon__lazy_171 nav__4174_43;
typedef struct anon__lazy_171 nav__4173_38;
typedef struct anon__lazy_114 nav__4173_41;
typedef struct anon__lazy_57 nav__4239_39;
typedef struct anon__lazy_289 nav__4239_40;
typedef struct anon__lazy_114 nav__4266_39;
typedef struct anon__lazy_299 nav__5112_38;
typedef struct anon__lazy_57 nav__5883_40;
typedef struct anon__lazy_114 nav__5883_51;
typedef struct anon__lazy_312 nav__1375_38;
typedef struct anon__lazy_57 nav__6010_45;
typedef struct anon__lazy_57 nav__6012_45;
typedef struct anon__lazy_171 nav__1363_38;
typedef struct anon__lazy_57 nav__1363_40;
typedef struct anon__lazy_171 nav__1355_38;
typedef struct anon__lazy_114 nav__1355_40;
typedef struct anon__lazy_171 nav__6016_38;
typedef struct anon__lazy_359 nav__6016_42;
struct anon__lazy_359 {
 uintptr_t f0;
 uint8_t f1;
};
typedef struct anon__lazy_125 nav__6017_39;
typedef struct anon__lazy_114 nav__6017_52;
typedef struct anon__lazy_171 nav__4171_38;
typedef struct anon__lazy_114 nav__4171_41;
typedef struct anon__lazy_57 nav__6018_39;
struct io_Writer__3628;
typedef struct anon__lazy_171 nav__6022_48;
typedef struct anon__lazy_57 nav__6022_50;
struct io_Writer__3628 {
 void const *context;
 struct anon__lazy_171 (*writeFn)(void const *, struct anon__lazy_57);
};
typedef struct anon__lazy_57 nav__6023_45;
typedef struct anon__lazy_57 nav__6024_45;
typedef struct anon__lazy_171 nav__4057_38;
typedef struct anon__lazy_57 nav__4057_41;
typedef struct anon__lazy_57 nav__6036_40;
typedef struct anon__lazy_171 nav__6036_43;
typedef struct anon__lazy_57 nav__6045_40;
typedef struct anon__lazy_171 nav__3911_38;
typedef struct anon__lazy_57 nav__3911_42;
typedef struct anon__lazy_171 nav__6035_38;
typedef struct anon__lazy_57 nav__6035_41;
typedef struct anon__lazy_57 nav__6046_40;
typedef struct anon__lazy_171 nav__6046_51;
typedef struct anon__lazy_57 nav__3905_41;
typedef struct anon__lazy_114 nav__3924_43;
typedef struct anon__lazy_171 nav__3924_52;
typedef struct anon__lazy_57 nav__3906_41;
typedef struct anon__lazy_114 nav__3906_47;
typedef struct anon__lazy_171 nav__3849_38;
typedef struct anon__lazy_359 nav__3849_42;
typedef struct anon__lazy_114 nav__3922_43;
typedef struct anon__lazy_114 nav__3923_43;
typedef struct anon__lazy_185 nav__3923_55;
typedef struct anon__lazy_114 nav__6132_39;
typedef struct anon__lazy_171 nav__6132_50;
typedef struct anon__lazy_114 nav__6133_39;
typedef struct anon__lazy_185 nav__6133_40;
typedef struct anon__lazy_125 nav__6133_51;
typedef struct anon__lazy_114 nav__6134_39;
typedef struct anon__lazy_114 nav__232_63;
typedef struct anon__lazy_114 nav__6135_52;
typedef struct anon__lazy_57 nav__341_58;
typedef struct anon__lazy_220 nav__341_66;
typedef struct anon__lazy_57 nav__342_58;
typedef struct anon__lazy_220 nav__342_66;
typedef struct anon__lazy_57 nav__343_56;
typedef struct anon__lazy_220 nav__343_66;
typedef struct anon__lazy_57 nav__6140_39;
typedef struct anon__lazy_419 nav__6142_45;
struct anon__lazy_419 {
 bool array[32];
};
typedef struct anon__lazy_421 nav__6142_47;
struct anon__lazy_421 {
 bool is_null;
 uint8_t payload;
};
typedef struct anon__lazy_421 nav__6508_38;
typedef struct anon__lazy_419 nav__6508_40;
typedef struct anon__lazy_57 nav__344_56;
typedef struct anon__lazy_220 nav__344_66;
typedef struct anon__lazy_435 nav__237_62;
struct anon__lazy_435 {
 uint8_t const *f0;
 uint8_t const *f1;
};
typedef struct anon__lazy_114 nav__237_65;
typedef struct anon__lazy_185 nav__237_67;
typedef struct anon__lazy_114 nav__6516_39;
typedef struct anon__lazy_185 nav__6516_40;
typedef struct anon__lazy_435 nav__6516_43;
typedef struct anon__lazy_114 nav__6517_40;
typedef struct anon__lazy_114 nav__6518_39;
typedef struct anon__lazy_185 nav__6518_40;
typedef struct anon__lazy_435 nav__6518_43;
typedef struct anon__lazy_289 nav__6518_54;
typedef struct anon__lazy_114 nav__6519_39;
typedef struct anon__lazy_435 nav__6561_40;
typedef struct anon__lazy_171 nav__6561_43;
typedef struct anon__lazy_57 nav__6561_44;
struct fmt_FormatOptions__4654;
typedef struct anon__lazy_289 nav__6561_55;
struct fmt_FormatOptions__4654 {
 struct anon__lazy_289 precision;
 struct anon__lazy_289 width;
 uint32_t fill;
 uint8_t alignment;
};
typedef struct anon__lazy_435 nav__6520_39;
struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302;
struct io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29__4294;
struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 {
 uint64_t bytes_written;
};
struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322;
struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *context;
};
typedef struct anon__lazy_171 nav__6520_53;
typedef struct anon__lazy_57 nav__6520_54;
typedef struct anon__lazy_289 nav__6521_38;
typedef struct anon__lazy_114 nav__6522_39;
typedef struct anon__lazy_185 nav__6522_40;
typedef struct anon__lazy_435 nav__6522_42;
struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712;
struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *context;
};
typedef struct anon__lazy_171 nav__6522_57;
typedef struct anon__lazy_57 nav__6522_58;
typedef struct anon__lazy_171 nav__6560_38;
typedef struct anon__lazy_57 nav__6560_41;
typedef struct anon__lazy_289 nav__6729_42;
typedef struct anon__lazy_171 nav__6729_46;
typedef struct anon__lazy_57 nav__6729_47;
typedef struct anon__lazy_171 nav__6742_38;
typedef struct anon__lazy_57 nav__6742_41;
typedef struct anon__lazy_114 nav__4130_39;
typedef struct anon__lazy_171 nav__6545_38;
typedef struct anon__lazy_57 nav__6545_42;
typedef struct anon__lazy_479 nav__6573_38;
struct anon__lazy_479 {
 uint16_t error;
 uint8_t payload;
};
typedef struct anon__lazy_171 nav__6589_38;
typedef struct anon__lazy_57 nav__6589_40;
typedef struct anon__lazy_479 nav__6589_48;
typedef struct anon__lazy_159 nav__6589_50;
typedef struct anon__lazy_57 nav__6743_39;
typedef struct anon__lazy_289 nav__6743_44;
typedef struct anon__lazy_171 nav__6743_48;
typedef struct anon__lazy_114 nav__6743_57;
typedef struct anon__lazy_479 nav__6743_60;
typedef struct anon__lazy_171 nav__4125_38;
typedef struct anon__lazy_57 nav__4125_42;
typedef struct anon__lazy_114 nav__4125_50;
typedef struct anon__lazy_171 nav__4035_38;
typedef struct anon__lazy_57 nav__4035_40;
typedef struct anon__lazy_159 nav__6579_38;
typedef struct anon__lazy_57 nav__6579_40;
typedef struct anon__lazy_494 nav__6579_48;
struct anon__lazy_494 {
 uint8_t array[2];
};
typedef struct anon__lazy_496 nav__6579_52;
struct anon__lazy_496 {
 uint8_t array[3];
};
typedef struct anon__lazy_498 nav__6579_56;
struct anon__lazy_498 {
 uint8_t array[4];
};
typedef struct anon__lazy_479 nav__6572_38;
typedef struct anon__lazy_479 nav__6744_38;
typedef struct anon__lazy_114 nav__6744_40;
typedef struct anon__lazy_479 nav__6574_38;
typedef struct anon__lazy_114 nav__6574_40;
typedef struct anon__lazy_57 nav__6040_40;
typedef struct anon__lazy_171 nav__6040_43;
typedef struct anon__lazy_159 nav__6581_38;
typedef struct anon__lazy_494 nav__6581_40;
typedef struct anon__lazy_159 nav__6583_38;
typedef struct anon__lazy_496 nav__6583_40;
typedef struct anon__lazy_159 nav__6587_38;
typedef struct anon__lazy_498 nav__6587_40;
typedef struct anon__lazy_159 nav__6585_38;
typedef struct anon__lazy_496 nav__6585_40;
struct Target_Os__563;
union Target_Os_VersionRange__619;
struct SemanticVersion_Range__625;
struct SemanticVersion__623;
typedef struct anon__lazy_57 nav__250_43;
struct SemanticVersion__623 {
 uintptr_t major;
 uintptr_t minor;
 uintptr_t patch;
 struct anon__lazy_57 pre;
 struct anon__lazy_57 build;
};
struct SemanticVersion_Range__625 {
 struct SemanticVersion__623 zig_e_min;
 struct SemanticVersion__623 zig_e_max;
};
struct Target_Os_HurdVersionRange__627;
struct Target_Os_HurdVersionRange__627 {
 struct SemanticVersion_Range__625 range;
 struct SemanticVersion__623 glibc;
};
struct Target_Os_LinuxVersionRange__629;
struct Target_Os_LinuxVersionRange__629 {
 struct SemanticVersion_Range__625 range;
 struct SemanticVersion__623 glibc;
 uint32_t android;
};
struct Target_Os_WindowsVersion_Range__685;
struct Target_Os_WindowsVersion_Range__685 {
 uint32_t zig_e_min;
 uint32_t zig_e_max;
};
union Target_Os_VersionRange__619 {
 struct SemanticVersion_Range__625 semver;
 struct Target_Os_HurdVersionRange__627 hurd;
 struct Target_Os_LinuxVersionRange__629 linux;
 struct Target_Os_WindowsVersion_Range__685 windows;
};
struct Target_Os__563 {
 union Target_Os_VersionRange__619 version_range;
 uint8_t tag;
};
struct Target_DynamicLinker__737;
struct Target_DynamicLinker__737 {
 uint8_t buffer[255];
 uint8_t len;
};
typedef struct anon__lazy_114 nav__3485_40;
typedef struct anon__lazy_114 nav__3486_40;
typedef struct anon__lazy_114 nav__3487_40;
typedef struct anon__lazy_114 nav__3449_46;
typedef struct anon__lazy_114 nav__230_46;
struct Target_Cpu_Feature_Set__754;
struct Target_Cpu_Feature_Set__754 {
 uintptr_t ints[5];
};
struct Target_Cpu__722;
struct Target_Cpu_Model__749;
struct Target_Cpu__722 {
 struct Target_Cpu_Model__749 const *model;
 struct Target_Cpu_Feature_Set__754 features;
 uint8_t arch;
};
typedef struct anon__lazy_57 nav__249_46;
struct Target_Cpu_Model__749 {
 struct anon__lazy_57 name;
 struct anon__lazy_57 llvm_name;
 struct Target_Cpu_Feature_Set__754 features;
};
struct Target__561;
typedef struct anon__lazy_57 nav__251_51;
struct Target__561 {
 struct Target_Cpu__722 cpu;
 struct Target_Os__563 os;
 uint8_t abi;
 uint8_t ofmt;
 struct Target_DynamicLinker__737 dynamic_linker;
};
struct builtin_CallingConvention__261;
struct builtin_CallingConvention_CommonOptions__263;
typedef struct anon__lazy_220 nav__830_40;
struct builtin_CallingConvention_CommonOptions__263 {
 struct anon__lazy_220 incoming_stack_alignment;
};
struct builtin_CallingConvention_X86RegparmOptions__265;
struct builtin_CallingConvention_X86RegparmOptions__265 {
 struct anon__lazy_220 incoming_stack_alignment;
 uint8_t register_params;
};
struct builtin_CallingConvention_ArmInterruptOptions__267;
struct builtin_CallingConvention_ArmInterruptOptions__267 {
 struct anon__lazy_220 incoming_stack_alignment;
 uint8_t type;
};
struct builtin_CallingConvention_MipsInterruptOptions__269;
struct builtin_CallingConvention_MipsInterruptOptions__269 {
 struct anon__lazy_220 incoming_stack_alignment;
 uint8_t mode;
};
struct builtin_CallingConvention_RiscvInterruptOptions__271;
struct builtin_CallingConvention_RiscvInterruptOptions__271 {
 struct anon__lazy_220 incoming_stack_alignment;
 uint8_t mode;
};
struct builtin_CallingConvention__261 {
 union {
  struct builtin_CallingConvention_CommonOptions__263 x86_64_sysv;
  struct builtin_CallingConvention_CommonOptions__263 x86_64_win;
  struct builtin_CallingConvention_CommonOptions__263 x86_64_regcall_v3_sysv;
  struct builtin_CallingConvention_CommonOptions__263 x86_64_regcall_v4_win;
  struct builtin_CallingConvention_CommonOptions__263 x86_64_vectorcall;
  struct builtin_CallingConvention_CommonOptions__263 x86_64_interrupt;
  struct builtin_CallingConvention_X86RegparmOptions__265 x86_sysv;
  struct builtin_CallingConvention_X86RegparmOptions__265 x86_win;
  struct builtin_CallingConvention_X86RegparmOptions__265 x86_stdcall;
  struct builtin_CallingConvention_CommonOptions__263 x86_fastcall;
  struct builtin_CallingConvention_CommonOptions__263 x86_thiscall;
  struct builtin_CallingConvention_CommonOptions__263 x86_thiscall_mingw;
  struct builtin_CallingConvention_CommonOptions__263 x86_regcall_v3;
  struct builtin_CallingConvention_CommonOptions__263 x86_regcall_v4_win;
  struct builtin_CallingConvention_CommonOptions__263 x86_vectorcall;
  struct builtin_CallingConvention_CommonOptions__263 x86_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 aarch64_aapcs;
  struct builtin_CallingConvention_CommonOptions__263 aarch64_aapcs_darwin;
  struct builtin_CallingConvention_CommonOptions__263 aarch64_aapcs_win;
  struct builtin_CallingConvention_CommonOptions__263 aarch64_vfabi;
  struct builtin_CallingConvention_CommonOptions__263 aarch64_vfabi_sve;
  struct builtin_CallingConvention_CommonOptions__263 arm_aapcs;
  struct builtin_CallingConvention_CommonOptions__263 arm_aapcs_vfp;
  struct builtin_CallingConvention_ArmInterruptOptions__267 arm_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 mips64_n64;
  struct builtin_CallingConvention_CommonOptions__263 mips64_n32;
  struct builtin_CallingConvention_MipsInterruptOptions__269 mips64_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 mips_o32;
  struct builtin_CallingConvention_MipsInterruptOptions__269 mips_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 riscv64_lp64;
  struct builtin_CallingConvention_CommonOptions__263 riscv64_lp64_v;
  struct builtin_CallingConvention_RiscvInterruptOptions__271 riscv64_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 riscv32_ilp32;
  struct builtin_CallingConvention_CommonOptions__263 riscv32_ilp32_v;
  struct builtin_CallingConvention_RiscvInterruptOptions__271 riscv32_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 sparc64_sysv;
  struct builtin_CallingConvention_CommonOptions__263 sparc_sysv;
  struct builtin_CallingConvention_CommonOptions__263 powerpc64_elf;
  struct builtin_CallingConvention_CommonOptions__263 powerpc64_elf_altivec;
  struct builtin_CallingConvention_CommonOptions__263 powerpc64_elf_v2;
  struct builtin_CallingConvention_CommonOptions__263 powerpc_sysv;
  struct builtin_CallingConvention_CommonOptions__263 powerpc_sysv_altivec;
  struct builtin_CallingConvention_CommonOptions__263 powerpc_aix;
  struct builtin_CallingConvention_CommonOptions__263 powerpc_aix_altivec;
  struct builtin_CallingConvention_CommonOptions__263 wasm_mvp;
  struct builtin_CallingConvention_CommonOptions__263 arc_sysv;
  struct builtin_CallingConvention_CommonOptions__263 bpf_std;
  struct builtin_CallingConvention_CommonOptions__263 csky_sysv;
  struct builtin_CallingConvention_CommonOptions__263 csky_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 hexagon_sysv;
  struct builtin_CallingConvention_CommonOptions__263 hexagon_sysv_hvx;
  struct builtin_CallingConvention_CommonOptions__263 lanai_sysv;
  struct builtin_CallingConvention_CommonOptions__263 loongarch64_lp64;
  struct builtin_CallingConvention_CommonOptions__263 loongarch32_ilp32;
  struct builtin_CallingConvention_CommonOptions__263 m68k_sysv;
  struct builtin_CallingConvention_CommonOptions__263 m68k_gnu;
  struct builtin_CallingConvention_CommonOptions__263 m68k_rtd;
  struct builtin_CallingConvention_CommonOptions__263 m68k_interrupt;
  struct builtin_CallingConvention_CommonOptions__263 msp430_eabi;
  struct builtin_CallingConvention_CommonOptions__263 propeller_sysv;
  struct builtin_CallingConvention_CommonOptions__263 s390x_sysv;
  struct builtin_CallingConvention_CommonOptions__263 s390x_sysv_vx;
  struct builtin_CallingConvention_CommonOptions__263 ve_sysv;
  struct builtin_CallingConvention_CommonOptions__263 xcore_xs1;
  struct builtin_CallingConvention_CommonOptions__263 xcore_xs2;
  struct builtin_CallingConvention_CommonOptions__263 xtensa_call0;
  struct builtin_CallingConvention_CommonOptions__263 xtensa_windowed;
  struct builtin_CallingConvention_CommonOptions__263 amdgcn_device;
  struct builtin_CallingConvention_CommonOptions__263 amdgcn_cs;
 } payload;
 uint8_t tag;
};
struct std_Options__3995;
typedef struct anon__lazy_289 nav__97_39;
struct std_Options__3995 {
 struct anon__lazy_289 page_size_min;
 struct anon__lazy_289 page_size_max;
 uintptr_t fmt_max_depth;
 bool enable_segfault_handler;
 uint8_t log_level;
 bool crypto_always_getrandom;
 bool crypto_fork_safety;
 bool keep_sigpipe;
 bool http_disable_tls;
 bool http_enable_ssl_key_log_file;
 uint8_t side_channels_mitigations;
};
typedef struct anon__lazy_289 nav__3465_38;
typedef struct anon__lazy_114 nav__3476_43;
typedef struct anon__lazy_57 nav__3508_40;
static uint8_t const __anon_1800[21];
static uint8_t const __anon_1860[21];
static uint8_t const __anon_2779[89];
static uint8_t const __anon_4395[10];
static uint8_t const __anon_4676[1];
static uint8_t const __anon_4696[1];
static uint8_t const __anon_4868[3];
static uint8_t const __anon_4742[4];
static uint8_t const __anon_4995[10];
static nav__987_39 fifo_UnixPipe_openWrite__987(struct mem_Allocator__206, nav__987_42);
static nav__986_39 fifo_UnixPipe_openRead__986(struct mem_Allocator__206, nav__986_42);
static nav__337_39 instruments_perf_PerfInstrument_init__337(struct mem_Allocator__206);
static nav__314_39 instruments_root_InstrumentHooks_init__314(struct mem_Allocator__206);
#define c_instrument_hooks_init__231 instrument_hooks_init
zig_extern struct instruments_root_InstrumentHooks__188 *instrument_hooks_init(void);
static nav__3488_40 mem_Allocator_create__anon_795__3488(struct mem_Allocator__206);
static struct instruments_valgrind_ValgrindInstrument__195 instruments_valgrind_ValgrindInstrument_init__325(struct mem_Allocator__206);
static uint16_t fs_accessAbsolute__1053(nav__1053_39, struct fs_File_OpenFlags__1807);
static nav__1050_39 fs_openFileAbsolute__1050(nav__1050_41, struct fs_File_OpenFlags__1807);
static struct fifo_UnixPipe_Writer__538 fifo_UnixPipe_Writer_init__988(struct fs_File__546, struct mem_Allocator__206);
static struct fifo_UnixPipe_Reader__540 fifo_UnixPipe_Reader_init__993(struct fs_File__546, struct mem_Allocator__206);
static uint16_t fifo_UnixPipe_Writer_sendCmd__991(struct fifo_UnixPipe_Writer__538 *, struct shared_Command__1919);
static nav__4169_38 bincode_deserializeInt__anon_2267__4169(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156);
static nav__4168_38 bincode_deserializeAlloc__anon_2256__4168(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4211_40 bincode_deserializePointerAlloc__anon_2393__4211(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4210_40 bincode_deserializeAlloc__anon_2380__4210(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4209_39 bincode_deserializeStructAlloc__anon_2355__4209(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4208_39 bincode_deserializeAlloc__anon_2326__4208(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static uint16_t bincode_deserializeAlloc__anon_2445__4214(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4216_39 bincode_deserializeStructAlloc__anon_2507__4216(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4215_39 bincode_deserializeAlloc__anon_2488__4215(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4167_39 bincode_deserializeUnionAlloc__anon_2251__4167(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__4166_39 bincode_deserializeAlloc__anon_2164__4166(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156, struct mem_Allocator__206);
static nav__996_39 fifo_UnixPipe_Reader_recvCmd__996(struct fifo_UnixPipe_Reader__540 *);
static nav__997_39 fifo_UnixPipe_Reader_waitForResponse__997(struct fifo_UnixPipe_Reader__540 *, nav__997_42);
static uint16_t fifo_UnixPipe_Reader_waitForAck__998(struct fifo_UnixPipe_Reader__540 *, nav__998_40);
static uint16_t instruments_perf_PerfInstrument_send_cmd__339(struct instruments_perf_PerfInstrument__200 *, struct shared_Command__1919);
static bool instruments_perf_PerfInstrument_is_instrumented__340(struct instruments_perf_PerfInstrument__200 *);
static nav__3737_39 mem_Allocator_allocBytesWithAlignment__anon_1889__3737(struct mem_Allocator__206, uintptr_t, uintptr_t);
static bool fs_path_isAbsolute__3772(nav__3772_39);
static void debug_assert__179(bool);
static struct fs_Dir__1809 fs_cwd__1045(void);
static uint16_t fs_Dir_access__3693(struct fs_Dir__1809, nav__3693_40, struct fs_File_OpenFlags__1807);
static nav__3629_39 fs_Dir_openFile__3629(struct fs_Dir__1809, nav__3629_42, struct fs_File_OpenFlags__1807);
static struct array_list_ArrayListAligned_28u8_2cnull_29__1950 array_list_ArrayListAligned_28u8_2cnull_29_init__3885(struct mem_Allocator__206);
static struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 array_list_ArrayListAligned_28u8_2cnull_29_writer__3910(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *);
static uint16_t bincode_serialize__anon_1991__4058(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__1919);
static void array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(struct array_list_ArrayListAligned_28u8_2cnull_29__1950);
static uint8_t const (*mem_asBytes__anon_2032__4059(uint32_t const *))[4];
static uint16_t fs_File_writeAll__1144(struct fs_File__546, nav__1144_40);
static zig_i128 time_nanoTimestamp__4072(void);
static nav__1134_38 fs_File_readAll__1134(struct fs_File__546, nav__1134_41);
static nav__4100_40 mem_Allocator_alloc__anon_2114__4100(struct mem_Allocator__206, uintptr_t);
static struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 io_fixed_buffer_stream_fixedBufferStream__anon_2136__4132(nav__4132_40);
static struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_reader__4121(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *);
static nav__4165_38 io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29_typeEras__4165(void const *, nav__4165_41);
static nav__4207_39 io_Reader_readBytesNoEof__anon_2313__4207(struct io_Reader__2282);
static nav__4212_39 io_Reader_readBytesNoEof__anon_2406__4212(struct io_Reader__2282);
static nav__4172_38 io_Reader_readAll__4172(struct io_Reader__2282, nav__4172_41);
static zig_cold zig_noreturn void bincode_invalidProtocol__anon_2430__4213(void);
static void mem_Allocator_free__anon_2541__4217(struct mem_Allocator__206, nav__4217_40);
static void utils_sleep__4222(uint64_t);
static void shared_Command_deinit__3835(struct shared_Command__1919, struct mem_Allocator__206);
static uint8_t mem_Alignment_fromByteUnits__628(uintptr_t);
static bool fs_path_isAbsolutePosix__3778(nav__3778_39);
static nav__1628_39 posix_toPosixPath__1628(nav__1628_41);
static uint16_t fs_Dir_accessZ__3694(struct fs_Dir__1809, uint8_t const *, struct fs_File_OpenFlags__1807);
static nav__3630_39 fs_Dir_openFileZ__3630(struct fs_Dir__1809, uint8_t const *, struct fs_File_OpenFlags__1807);
static uint16_t bincode_serializeUnion__anon_2692__4236(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__1919);
static nav__3933_39 array_list_ArrayListAligned_28u8_2cnull_29_allocatedSlice__3933(struct array_list_ArrayListAligned_28u8_2cnull_29__1950);
static nav__1143_38 fs_File_write__1143(struct fs_File__546, nav__1143_41);
static nav__1544_39 posix_clock_gettime__1544(uint32_t);
static nav__1133_38 fs_File_read__1133(struct fs_File__546, nav__1133_41);
static nav__4237_39 mem_Allocator_allocWithSizeAndAlignment__anon_2759__4237(struct mem_Allocator__206, uintptr_t, uintptr_t);
static nav__4124_38 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_read__4124(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *, nav__4124_42);
static uint16_t io_Reader_readNoEof__4174(struct io_Reader__2282, nav__4174_40);
static nav__4173_38 io_Reader_readAtLeast__4173(struct io_Reader__2282, nav__4173_41, uintptr_t);
static zig_cold zig_noreturn void debug_no_panic_call__4239(nav__4239_39, nav__4239_40);
static nav__4266_39 mem_sliceAsBytes__anon_2876__4266(nav__4266_39);
static nav__5112_38 math_cast__anon_2915__5112(uint64_t);
static void mem_Allocator_free__anon_2982__5883(struct mem_Allocator__206, nav__5883_40);
static bool math_isPowerOfTwo__anon_2984__5884(uintptr_t);
static uint16_t posix_faccessatZ__1519(int32_t, uint8_t const *, uint32_t, uint32_t);
static nav__1375_38 posix_openatZ__1375(int32_t, uint8_t const *, uint32_t, uintptr_t);
static uint16_t posix_flock__1536(int32_t, int32_t);
static void posix_close__1333(int32_t);
static uint16_t bincode_serialize__anon_3123__6009(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, uint32_t);
static uint16_t bincode_serialize__anon_3125__6010(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__struct_1922__1922);
static uint16_t bincode_serialize__anon_3127__6011(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981);
static uint16_t bincode_serialize__anon_3129__6012(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__struct_1923__1923);
static nav__1363_38 posix_write__1363(int32_t, nav__1363_40);
static uint16_t posix_errno__anon_3293__6015(int);
static uint16_t posix_unexpectedErrno__1627(uint16_t);
static nav__1355_38 posix_read__1355(int32_t, nav__1355_40);
static nav__6016_38 math_mul__anon_3314__6016(uintptr_t, uintptr_t);
static nav__6017_39 mem_Allocator_allocBytesWithAlignment__anon_3316__6017(struct mem_Allocator__206, uintptr_t, uintptr_t);
static nav__4171_38 io_Reader_read__4171(struct io_Reader__2282, nav__4171_41);
static nav__6018_39 mem_sliceAsBytes__anon_3347__6018(nav__6018_39);
static uint16_t bincode_serializeInt__anon_3509__6022(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, uint32_t);
static uint16_t bincode_serializeStruct__anon_3514__6023(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__struct_1922__1922);
static uint16_t bincode_serializeStruct__anon_3515__6024(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, struct shared_Command__struct_1923__1923);
static uint16_t posix_errno__anon_3526__6026(intptr_t);
static nav__4057_38 io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29___4057(void const *, nav__4057_41);
static uint16_t io_Writer_writeAll__6036(struct io_Writer__3628, nav__6036_40);
static uint16_t bincode_serialize__anon_3664__6045(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, nav__6045_40);
static nav__3911_38 array_list_ArrayListAligned_28u8_2cnull_29_appendWrite__3911(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, nav__3911_42);
static nav__6035_38 io_Writer_write__6035(struct io_Writer__3628, nav__6035_41);
static uint16_t bincode_serializePointer__anon_3676__6046(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981, nav__6046_40);
static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_appendSlice__3905(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, nav__3905_41);
static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureUnusedCapacity__3924(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, uintptr_t);
static void array_list_ArrayListAligned_28u8_2cnull_29_appendSliceAssumeCapacity__3906(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, nav__3906_41);
static nav__3849_38 array_list_addOrOom__3849(uintptr_t, uintptr_t);
static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacity__3922(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, uintptr_t);
static uintptr_t array_list_ArrayListAlignedUnmanaged_28u8_2cnull_29_growCapacity__6105(uintptr_t, uintptr_t);
static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacityPrecise__3923(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *, uintptr_t);
static nav__6132_39 mem_Allocator_remap__anon_3736__6132(struct mem_Allocator__206, nav__6132_39, uintptr_t);
static nav__6133_40 mem_Allocator_alignedAlloc__anon_3741__6133(struct mem_Allocator__206, uintptr_t);
static nav__6134_39 mem_bytesAsSlice__anon_3756__6134(nav__6134_39);
#define c_instrument_hooks_deinit__232 instrument_hooks_deinit
zig_extern void instrument_hooks_deinit(struct instruments_root_InstrumentHooks__188 *);
static void instruments_perf_PerfInstrument_deinit__338(struct instruments_perf_PerfInstrument__200 *);
static void mem_Allocator_destroy__anon_3774__6135(struct mem_Allocator__206, struct instruments_root_InstrumentHooks__188 *);
static void fifo_UnixPipe_Writer_deinit__992(struct fifo_UnixPipe_Writer__538 *);
static void fifo_UnixPipe_Reader_deinit__999(struct fifo_UnixPipe_Reader__540 *);
static void fs_File_close__1090(struct fs_File__546);
#define c_instrument_hooks_is_instrumented__233 instrument_hooks_is_instrumented
zig_extern bool instrument_hooks_is_instrumented(struct instruments_root_InstrumentHooks__188 *);
static zig_cold uint16_t instruments_perf_PerfInstrument_start_benchmark__341(struct instruments_perf_PerfInstrument__200 *);
#define c_instrument_hooks_start_benchmark__234 instrument_hooks_start_benchmark
zig_extern uint8_t instrument_hooks_start_benchmark(struct instruments_root_InstrumentHooks__188 *);
static zig_cold uint16_t instruments_perf_PerfInstrument_stop_benchmark__342(struct instruments_perf_PerfInstrument__200 *);
#define c_instrument_hooks_stop_benchmark__235 instrument_hooks_stop_benchmark
zig_extern uint8_t instrument_hooks_stop_benchmark(struct instruments_root_InstrumentHooks__188 *);
static uint16_t instruments_perf_PerfInstrument_set_executed_benchmark__343(struct instruments_perf_PerfInstrument__200 *, uint32_t, uint8_t const *);
#define c_instrument_hooks_executed_benchmark__236 instrument_hooks_executed_benchmark
zig_extern uint8_t instrument_hooks_executed_benchmark(struct instruments_root_InstrumentHooks__188 *, uint32_t, uint8_t const *);
static nav__6140_39 mem_span__anon_3885__6140(uint8_t const *);
static uintptr_t mem_len__anon_3894__6141(uint8_t const *);
static uintptr_t mem_indexOfSentinel__anon_3902__6142(uint8_t const *);
static bool math_isPowerOfTwo__anon_3913__6143(void);
static nav__6508_38 simd_firstTrue__anon_4125__6508(nav__6508_40);
static uintptr_t mem_alignForward__anon_4128__6509(uintptr_t, uintptr_t);
static bool mem_isAligned__499(uintptr_t, uintptr_t);
static bool mem_isValidAlignGeneric__anon_4186__6510(uintptr_t);
static uintptr_t mem_alignBackward__anon_4187__6511(uintptr_t, uintptr_t);
static bool mem_isAlignedGeneric__anon_4193__6512(uint64_t, uint64_t);
static uint64_t mem_alignBackward__anon_4195__6513(uint64_t, uint64_t);
static bool mem_isValidAlignGeneric__anon_4197__6514(uint64_t);
static bool math_isPowerOfTwo__anon_4198__6515(uint64_t);
static uint16_t instruments_perf_PerfInstrument_set_integration__344(struct instruments_perf_PerfInstrument__200 *, uint8_t const *, uint8_t const *);
#define c_instrument_hooks_set_integration__237 instrument_hooks_set_integration
zig_extern uint8_t instrument_hooks_set_integration(struct instruments_root_InstrumentHooks__188 *, uint8_t const *, uint8_t const *);
static nav__6516_40 fmt_allocPrintZ__anon_4232__6516(struct mem_Allocator__206, nav__6516_43);
static void mem_Allocator_free__anon_4234__6517(struct mem_Allocator__206, nav__6517_40);
static nav__6518_40 fmt_allocPrint__anon_4259__6518(struct mem_Allocator__206, nav__6518_43);
static nav__6519_39 mem_sliceAsBytes__anon_4266__6519(nav__6519_39);
static uint16_t fmt_format__anon_4341__6561(struct io_Writer__3628, nav__6561_40);
static uint64_t fmt_count__anon_4273__6520(nav__6520_39);
static nav__6521_38 math_cast__anon_4275__6521(uint64_t);
static nav__6522_40 fmt_bufPrint__anon_4284__6522(nav__6522_39, nav__6522_42);
static struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 io_counting_writer_countingWriter__anon_4305__6547(void);
static struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_writer__6546(struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *);
static nav__6560_38 io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWr__6560(void const *, nav__6560_41);
static uint16_t fmt_formatType__anon_4668__6729(uint8_t const *, struct fmt_FormatOptions__4654, struct io_Writer__3628, uintptr_t);
static struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_writer__4122(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *);
static nav__6742_38 io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write__6742(void const *, nav__6742_41);
static nav__4130_39 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_getWritten__4130(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133);
static nav__6545_38 io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_write__6545(struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *, nav__6545_42);
static nav__6573_38 unicode_utf8ByteSequenceLength__6573(uint8_t);
static nav__6589_38 unicode_utf8CountCodepoints__6589(nav__6589_40);
static uint16_t fmt_formatBuf__anon_4769__6743(nav__6743_39, struct fmt_FormatOptions__4654, struct io_Writer__3628);
static nav__4125_38 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_write__4125(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *, nav__4125_42);
static nav__4035_38 io_dummyWrite__4035(nav__4035_40);
static nav__6579_38 unicode_utf8Decode__6579(nav__6579_40);
static nav__6572_38 unicode_utf8CodepointSequenceLength__6572(uint32_t);
static nav__6744_38 unicode_utf8EncodeImpl__anon_4900__6744(uint32_t, nav__6744_40);
static nav__6574_38 unicode_utf8Encode__6574(uint32_t, nav__6574_40);
static uint16_t io_Writer_writeBytesNTimes__6040(struct io_Writer__3628, nav__6040_40, uintptr_t);
static nav__6581_38 unicode_utf8Decode2__6581(nav__6581_40);
static nav__6583_38 unicode_utf8Decode3__6583(nav__6583_40);
static nav__6587_38 unicode_utf8Decode4__6587(nav__6587_40);
static bool unicode_isSurrogateCodepoint__6652(uint32_t);
static nav__6585_38 unicode_utf8Decode3AllowSurrogateHalf__6585(nav__6585_40);
static uint64_t const builtin_zig_backend__242;
static bool const start_simplified_logic__109;
static uint8_t const builtin_output_mode__243;
static uint8_t const builtin_link_mode__244;
static bool const builtin_link_libc__254;
static bool const posix_use_libc__1196;
static struct Target_Os__563 const builtin_os__250;
static uint8_t const c_native_os__1638;
static struct Target_DynamicLinker__737 const Target_DynamicLinker_none__3360;
static bool const builtin_is_test__246;
static uint8_t *heap_CAllocator_alloc__3484(void *, uintptr_t, uint8_t, uintptr_t);
static bool heap_CAllocator_resize__3485(void *, nav__3485_40, uint8_t, uintptr_t, uintptr_t);
static uint8_t *heap_CAllocator_remap__3486(void *, nav__3486_40, uint8_t, uintptr_t, uintptr_t);
static void heap_CAllocator_free__3487(void *, nav__3487_40, uint8_t, uintptr_t);
static uint8_t *heap_CAllocator_alignedAlloc__3481(uintptr_t, uint8_t);
static uintptr_t heap_CAllocator_alignedAllocSize__3483(uint8_t *);
static void heap_CAllocator_alignedFree__3482(uint8_t *);
static uintptr_t mem_Alignment_toByteUnits__627(uint8_t);
static struct mem_Allocator__206 const heap_c_allocator__3449;
static struct mem_Allocator__206 const c_allocator__230;
static struct Target_Cpu_Feature_Set__754 const Target_Cpu_Feature_Set_empty__3405;
static struct Target_Cpu__722 const builtin_cpu__249;
static uint8_t const builtin_abi__248;
static uint8_t const builtin_object_format__252;
static struct Target__561 const builtin_target__251;
static struct builtin_CallingConvention__261 const builtin_CallingConvention_c__830;
zig_extern uint8_t running_on_valgrind(void);
static uint8_t const (*const shared_RUNNER_CTL_FIFO__3612)[21];
static uint8_t const (*const shared_RUNNER_ACK_FIFO__3613)[21];
static uint8_t const mem_native_endian__353;
static uint8_t const fs_path_native_os__3751;
static uint8_t const fs_native_os__1008;
static uint8_t const fs_Dir_native_os__3735;
static uint8_t const os_linux_native_arch__2544;
zig_extern int nanosleep(struct cimport_struct_timespec__2902 const *, struct cimport_struct_timespec__2902 *);
static int const cimport_EINTR__5751;
static uint8_t const builtin_mode__253;
static bool const debug_runtime_safety__159;
zig_extern int flock(int32_t, int);
static bool const fs_Dir_have_flock__3736;
static bool const fs_File_is_windows__1184;
static uint8_t const posix_native_os__1194;
zig_extern int clock_gettime(uint32_t, struct os_linux_timespec__struct_2728__2728 *);
zig_extern int faccessat(int32_t, uint8_t const *, unsigned int, unsigned int);
static bool const posix_lfs64_abi__1624;
zig_extern int openat64(int, uint8_t const *, uint32_t, ...);
zig_extern int close(int32_t);
zig_extern intptr_t write(int32_t, uint8_t const *, uintptr_t);
static uint8_t const c_native_abi__1636;
zig_extern int *zig_e___errno_location(void) zig_mangled(zig_e___errno_location, "__errno_location");
static bool const posix_unexpected_error_tracing__1625;
zig_extern intptr_t read(int32_t, uint8_t *, uintptr_t);
zig_extern void callgrind_zero_stats(void);
zig_extern void callgrind_start_instrumentation(void);
zig_extern void callgrind_stop_instrumentation(void);
zig_extern void callgrind_dump_stats_at(uint8_t const *);
static bool const mem_backend_supports_vectors__381;
static bool const builtin_valgrind_support__257;
static bool const debug_default_enable_segfault_handler__204;
static uint8_t const log_default_level__6277;
static struct std_Options__3995 const std_options__97;
static nav__3465_38 const heap_page_size_min_default__3465;
static uintptr_t const heap_page_size_min__3444;
static uint16_t const fmt_max_format_args__6345;
static uint8_t const (*const fmt_ANY__6348)[4];
static uint8_t const unicode_native_endian__6570;
static uint32_t const unicode_replacement_character__6571;
static bool const os_linux_extern_getauxval__2608;
static struct mem_Allocator_VTable__209 const heap_CAllocator_vtable__3476;
zig_extern uintptr_t malloc_usable_size(void const *);
static bool const heap_CAllocator_supports_malloc_size__3477;
static bool const heap_CAllocator_supports_posix_memalign__3479;
zig_extern int posix_memalign(void **, uintptr_t, uintptr_t);
zig_extern void free(void *);
static struct Target_Cpu_Model__749 const Target_x86_cpu_alderlake__3508;
enum {
 zig_error_OutOfMemory = 1u,
 zig_error_PermissionDenied = 2u,
 zig_error_FileNotFound = 3u,
 zig_error_NameTooLong = 4u,
 zig_error_InputOutput = 5u,
 zig_error_SystemResources = 6u,
 zig_error_BadPathName = 7u,
 zig_error_FileBusy = 8u,
 zig_error_SymLinkLoop = 9u,
 zig_error_ReadOnlyFileSystem = 10u,
 zig_error_InvalidUtf8 = 11u,
 zig_error_InvalidWtf8 = 12u,
 zig_error_Unexpected = 13u,
 zig_error_SharingViolation = 14u,
 zig_error_PathAlreadyExists = 15u,
 zig_error_AccessDenied = 16u,
 zig_error_PipeBusy = 17u,
 zig_error_NoDevice = 18u,
 zig_error_NetworkNotFound = 19u,
 zig_error_AntivirusInterference = 20u,
 zig_error_ProcessFdQuotaExceeded = 21u,
 zig_error_SystemFdQuotaExceeded = 22u,
 zig_error_FileTooBig = 23u,
 zig_error_IsDir = 24u,
 zig_error_NoSpaceLeft = 25u,
 zig_error_NotDir = 26u,
 zig_error_DeviceBusy = 27u,
 zig_error_FileLocksNotSupported = 28u,
 zig_error_WouldBlock = 29u,
 zig_error_DiskQuota = 30u,
 zig_error_InvalidArgument = 31u,
 zig_error_BrokenPipe = 32u,
 zig_error_OperationAborted = 33u,
 zig_error_NotOpenForWriting = 34u,
 zig_error_LockViolation = 35u,
 zig_error_ConnectionResetByPeer = 36u,
 zig_error_ProcessNotFound = 37u,
 zig_error_AckTimeout = 38u,
 zig_error_ConnectionTimedOut = 39u,
 zig_error_NotOpenForReading = 40u,
 zig_error_SocketNotConnected = 41u,
 zig_error_Canceled = 42u,
 zig_error_EndOfStream = 43u,
 zig_error_UnexpectedError = 44u,
 zig_error_UnexpectedResponse = 45u,
 zig_error_UnsupportedClock = 46u,
 zig_error_Overflow = 47u,
 zig_error_Utf8ExpectedContinuation = 48u,
 zig_error_Utf8OverlongEncoding = 49u,
 zig_error_Utf8EncodesSurrogateHalf = 50u,
 zig_error_Utf8CodepointTooLarge = 51u,
 zig_error_Utf8InvalidStartByte = 52u,
 zig_error_TruncatedInput = 53u,
 zig_error_Utf8CannotEncodeSurrogateHalf = 54u,
 zig_error_CodepointTooLarge = 55u,
};
static uint8_t const zig_errorName_OutOfMemory[12] = "OutOfMemory";
static uint8_t const zig_errorName_PermissionDenied[17] = "PermissionDenied";
static uint8_t const zig_errorName_FileNotFound[13] = "FileNotFound";
static uint8_t const zig_errorName_NameTooLong[12] = "NameTooLong";
static uint8_t const zig_errorName_InputOutput[12] = "InputOutput";
static uint8_t const zig_errorName_SystemResources[16] = "SystemResources";
static uint8_t const zig_errorName_BadPathName[12] = "BadPathName";
static uint8_t const zig_errorName_FileBusy[9] = "FileBusy";
static uint8_t const zig_errorName_SymLinkLoop[12] = "SymLinkLoop";
static uint8_t const zig_errorName_ReadOnlyFileSystem[19] = "ReadOnlyFileSystem";
static uint8_t const zig_errorName_InvalidUtf8[12] = "InvalidUtf8";
static uint8_t const zig_errorName_InvalidWtf8[12] = "InvalidWtf8";
static uint8_t const zig_errorName_Unexpected[11] = "Unexpected";
static uint8_t const zig_errorName_SharingViolation[17] = "SharingViolation";
static uint8_t const zig_errorName_PathAlreadyExists[18] = "PathAlreadyExists";
static uint8_t const zig_errorName_AccessDenied[13] = "AccessDenied";
static uint8_t const zig_errorName_PipeBusy[9] = "PipeBusy";
static uint8_t const zig_errorName_NoDevice[9] = "NoDevice";
static uint8_t const zig_errorName_NetworkNotFound[16] = "NetworkNotFound";
static uint8_t const zig_errorName_AntivirusInterference[22] = "AntivirusInterference";
static uint8_t const zig_errorName_ProcessFdQuotaExceeded[23] = "ProcessFdQuotaExceeded";
static uint8_t const zig_errorName_SystemFdQuotaExceeded[22] = "SystemFdQuotaExceeded";
static uint8_t const zig_errorName_FileTooBig[11] = "FileTooBig";
static uint8_t const zig_errorName_IsDir[6] = "IsDir";
static uint8_t const zig_errorName_NoSpaceLeft[12] = "NoSpaceLeft";
static uint8_t const zig_errorName_NotDir[7] = "NotDir";
static uint8_t const zig_errorName_DeviceBusy[11] = "DeviceBusy";
static uint8_t const zig_errorName_FileLocksNotSupported[22] = "FileLocksNotSupported";
static uint8_t const zig_errorName_WouldBlock[11] = "WouldBlock";
static uint8_t const zig_errorName_DiskQuota[10] = "DiskQuota";
static uint8_t const zig_errorName_InvalidArgument[16] = "InvalidArgument";
static uint8_t const zig_errorName_BrokenPipe[11] = "BrokenPipe";
static uint8_t const zig_errorName_OperationAborted[17] = "OperationAborted";
static uint8_t const zig_errorName_NotOpenForWriting[18] = "NotOpenForWriting";
static uint8_t const zig_errorName_LockViolation[14] = "LockViolation";
static uint8_t const zig_errorName_ConnectionResetByPeer[22] = "ConnectionResetByPeer";
static uint8_t const zig_errorName_ProcessNotFound[16] = "ProcessNotFound";
static uint8_t const zig_errorName_AckTimeout[11] = "AckTimeout";
static uint8_t const zig_errorName_ConnectionTimedOut[19] = "ConnectionTimedOut";
static uint8_t const zig_errorName_NotOpenForReading[18] = "NotOpenForReading";
static uint8_t const zig_errorName_SocketNotConnected[19] = "SocketNotConnected";
static uint8_t const zig_errorName_Canceled[9] = "Canceled";
static uint8_t const zig_errorName_EndOfStream[12] = "EndOfStream";
static uint8_t const zig_errorName_UnexpectedError[16] = "UnexpectedError";
static uint8_t const zig_errorName_UnexpectedResponse[19] = "UnexpectedResponse";
static uint8_t const zig_errorName_UnsupportedClock[17] = "UnsupportedClock";
static uint8_t const zig_errorName_Overflow[9] = "Overflow";
static uint8_t const zig_errorName_Utf8ExpectedContinuation[25] = "Utf8ExpectedContinuation";
static uint8_t const zig_errorName_Utf8OverlongEncoding[21] = "Utf8OverlongEncoding";
static uint8_t const zig_errorName_Utf8EncodesSurrogateHalf[25] = "Utf8EncodesSurrogateHalf";
static uint8_t const zig_errorName_Utf8CodepointTooLarge[22] = "Utf8CodepointTooLarge";
static uint8_t const zig_errorName_Utf8InvalidStartByte[21] = "Utf8InvalidStartByte";
static uint8_t const zig_errorName_TruncatedInput[15] = "TruncatedInput";
static uint8_t const zig_errorName_Utf8CannotEncodeSurrogateHalf[30] = "Utf8CannotEncodeSurrogateHalf";
static uint8_t const zig_errorName_CodepointTooLarge[18] = "CodepointTooLarge";
static struct anon__lazy_57 const zig_errorName[56] = {{zig_errorName_OutOfMemory, 11ul}, {zig_errorName_PermissionDenied, 16ul}, {zig_errorName_FileNotFound, 12ul}, {zig_errorName_NameTooLong, 11ul}, {zig_errorName_InputOutput, 11ul}, {zig_errorName_SystemResources, 15ul}, {zig_errorName_BadPathName, 11ul}, {zig_errorName_FileBusy, 8ul}, {zig_errorName_SymLinkLoop, 11ul}, {zig_errorName_ReadOnlyFileSystem, 18ul}, {zig_errorName_InvalidUtf8, 11ul}, {zig_errorName_InvalidWtf8, 11ul}, {zig_errorName_Unexpected, 10ul}, {zig_errorName_SharingViolation, 16ul}, {zig_errorName_PathAlreadyExists, 17ul}, {zig_errorName_AccessDenied, 12ul}, {zig_errorName_PipeBusy, 8ul}, {zig_errorName_NoDevice, 8ul}, {zig_errorName_NetworkNotFound, 15ul}, {zig_errorName_AntivirusInterference, 21ul}, {zig_errorName_ProcessFdQuotaExceeded, 22ul}, {zig_errorName_SystemFdQuotaExceeded, 21ul}, {zig_errorName_FileTooBig, 10ul}, {zig_errorName_IsDir, 5ul}, {zig_errorName_NoSpaceLeft, 11ul}, {zig_errorName_NotDir, 6ul}, {zig_errorName_DeviceBusy, 10ul}, {zig_errorName_FileLocksNotSupported, 21ul}, {zig_errorName_WouldBlock, 10ul}, {zig_errorName_DiskQuota, 9ul}, {zig_errorName_InvalidArgument, 15ul}, {zig_errorName_BrokenPipe, 10ul}, {zig_errorName_OperationAborted, 16ul}, {zig_errorName_NotOpenForWriting, 17ul}, {zig_errorName_LockViolation, 13ul}, {zig_errorName_ConnectionResetByPeer, 21ul}, {zig_errorName_ProcessNotFound, 15ul}, {zig_errorName_AckTimeout, 10ul}, {zig_errorName_ConnectionTimedOut, 18ul}, {zig_errorName_NotOpenForReading, 17ul}, {zig_errorName_SocketNotConnected, 18ul}, {zig_errorName_Canceled, 8ul}, {zig_errorName_EndOfStream, 11ul}, {zig_errorName_UnexpectedError, 15ul}, {zig_errorName_UnexpectedResponse, 18ul}, {zig_errorName_UnsupportedClock, 16ul}, {zig_errorName_Overflow, 8ul}, {zig_errorName_Utf8ExpectedContinuation, 24ul}, {zig_errorName_Utf8OverlongEncoding, 20ul}, {zig_errorName_Utf8EncodesSurrogateHalf, 24ul}, {zig_errorName_Utf8CodepointTooLarge, 21ul}, {zig_errorName_Utf8InvalidStartByte, 20ul}, {zig_errorName_TruncatedInput, 14ul}, {zig_errorName_Utf8CannotEncodeSurrogateHalf, 29ul}, {zig_errorName_CodepointTooLarge, 17ul}};

static uint8_t const __anon_1800[21] = "/tmp/runner.ctl.fifo";

static uint8_t const __anon_1860[21] = "/tmp/runner.ack.fifo";

static uint8_t const __anon_2779[89] = "Invalid protocol detected: The stream end was found before all required bytes were read.";

static uint8_t const __anon_4395[10] = "Metadata: ";

static uint8_t const __anon_4676[1] = " ";

static uint8_t const __anon_4696[1] = "\000";

static uint8_t const __anon_4868[3] = "\357\277\275";

static uint8_t const __anon_4742[4] = "any";

static uint8_t const __anon_4995[10] = "alderlake";

static nav__987_39 fifo_UnixPipe_openWrite__987(struct mem_Allocator__206 const a0, nav__987_42 const a1) {
 nav__987_39 t1;
 struct fifo_UnixPipe_Writer__538 t4;
 nav__987_56 t2;
 struct fs_File__546 t3;
 uint16_t t0;
 t0 = fs_accessAbsolute__1053(a1, (struct fs_File_OpenFlags__1807){UINT8_C(2),UINT8_C(0),false,false});
 if (t0) {
  t1.payload = (struct fifo_UnixPipe_Writer__538){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}};
  t1.error = t0;
  return t1;
 }
 t2 = fs_openFileAbsolute__1050(a1, (struct fs_File_OpenFlags__1807){UINT8_C(2),UINT8_C(1),true,false});
 if (t2.error) {
  t0 = t2.error;
  t1.payload = (struct fifo_UnixPipe_Writer__538){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}};
  t1.error = t0;
  return t1;
 }
 t3 = t2.payload;
 t4 = fifo_UnixPipe_Writer_init__988(t3, a0);
 t1.payload = t4;
 t1.error = UINT16_C(0);
 return t1;
}

static nav__986_39 fifo_UnixPipe_openRead__986(struct mem_Allocator__206 const a0, nav__986_42 const a1) {
 nav__986_39 t1;
 struct fifo_UnixPipe_Reader__540 t4;
 nav__986_56 t2;
 struct fs_File__546 t3;
 uint16_t t0;
 t0 = fs_accessAbsolute__1053(a1, (struct fs_File_OpenFlags__1807){UINT8_C(2),UINT8_C(0),false,false});
 if (t0) {
  t1.payload = (struct fifo_UnixPipe_Reader__540){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}};
  t1.error = t0;
  return t1;
 }
 t2 = fs_openFileAbsolute__1050(a1, (struct fs_File_OpenFlags__1807){UINT8_C(2),UINT8_C(1),true,false});
 if (t2.error) {
  t0 = t2.error;
  t1.payload = (struct fifo_UnixPipe_Reader__540){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}};
  t1.error = t0;
  return t1;
 }
 t3 = t2.payload;
 t4 = fifo_UnixPipe_Reader_init__993(t3, a0);
 t1.payload = t4;
 t1.error = UINT16_C(0);
 return t1;
}

static nav__337_39 instruments_perf_PerfInstrument_init__337(struct mem_Allocator__206 const a0) {
 struct instruments_perf_PerfInstrument__200 *t1;
 struct mem_Allocator__206 *t2;
 struct fifo_UnixPipe_Writer__538 *t3;
 nav__337_61 t4;
 nav__337_39 t6;
 nav__337_39 t0;
 struct fifo_UnixPipe_Writer__538 t7;
 struct fifo_UnixPipe_Reader__540 *t8;
 nav__337_66 t9;
 struct fifo_UnixPipe_Reader__540 t10;
 uint16_t t5;
 t0.error = UINT16_C(0);
 t1 = &t0.payload;
 t2 = (struct mem_Allocator__206 *)&t1->allocator;
 (*t2) = a0;
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t1->writer;
 t4 = fifo_UnixPipe_openWrite__987(a0, (nav__337_59){(uint8_t const *)&__anon_1800,(uintptr_t)20ul});
 if (t4.error) {
  t5 = t4.error;
  t6.payload = (struct instruments_perf_PerfInstrument__200){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}},{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}}};
  t6.error = t5;
  return t6;
 }
 t7 = t4.payload;
 (*t3) = t7;
 t8 = (struct fifo_UnixPipe_Reader__540 *)&t1->reader;
 t9 = fifo_UnixPipe_openRead__986(a0, (nav__337_59){(uint8_t const *)&__anon_1860,(uintptr_t)20ul});
 if (t9.error) {
  t5 = t9.error;
  t6.payload = (struct instruments_perf_PerfInstrument__200){{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}},{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)},{-INT32_C(0x55555556)}}};
  t6.error = t5;
  return t6;
 }
 t10 = t9.payload;
 (*t8) = t10;
 return t0;
}

static nav__314_39 instruments_root_InstrumentHooks_init__314(struct mem_Allocator__206 const a0) {
 struct instruments_valgrind_ValgrindInstrument__195 t2;
 struct instruments_root_InstrumentHooks__188 t3;
 nav__314_39 t4;
 struct instruments_perf_PerfInstrument__200 t6;
 struct instruments_perf_PerfInstrument__200 t8;
 struct instruments_perf_PerfInstrument__200 t5;
 nav__314_60 t7;
 uint8_t t0;
 bool t1;
 t0 = running_on_valgrind();
 t1 = t0 > UINT8_C(0);
 if (t1) {
  t2 = instruments_valgrind_ValgrindInstrument_init__325(a0);
  t3.tag = UINT8_C(0);
  t3.payload.valgrind = t2;
  t4.payload = t3;
  t4.error = UINT16_C(0);
  return t4;
 }
 goto zig_block_0;

 zig_block_0:;
 t7 = instruments_perf_PerfInstrument_init__337(a0);
 t1 = t7.error == UINT16_C(0);
 if (t1) {
  t8 = t7.payload;
  t6 = t8;
  goto zig_block_1;
 }
 return (nav__314_39){{{{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)}}},UINT8_C(2)},0};

 zig_block_1:;
 t5 = t6;
 t1 = instruments_perf_PerfInstrument_is_instrumented__340(&t5);
 if (t1) {
  t6 = t5;
  t3.tag = UINT8_C(1);
  t3.payload.perf = t6;
  t4.payload = t3;
  t4.error = UINT16_C(0);
  return t4;
 }
 goto zig_block_2;

 zig_block_2:;
 return (nav__314_39){{{{{((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul)}}},UINT8_C(2)},0};
}

struct instruments_root_InstrumentHooks__188 *c_instrument_hooks_init__231(void) {
 struct instruments_root_InstrumentHooks__188 *t0;
 struct instruments_root_InstrumentHooks__188 *t3;
 nav__231_46 t1;
 nav__231_71 t4;
 struct instruments_root_InstrumentHooks__188 t5;
 bool t2;
 t1 = mem_Allocator_create__anon_795__3488((struct mem_Allocator__206){((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)&heap_CAllocator_vtable__3476)});
 t2 = t1.error == UINT16_C(0);
 if (t2) {
  t3 = t1.payload;
  t0 = t3;
  goto zig_block_0;
 }
 return NULL;

 zig_block_0:;
 t4 = instruments_root_InstrumentHooks_init__314((struct mem_Allocator__206){((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)&heap_CAllocator_vtable__3476)});
 t5 = t4.payload;
 (*t0) = t5;
 t3 = (struct instruments_root_InstrumentHooks__188 *)t0;
 return t3;
}

static nav__3488_40 mem_Allocator_create__anon_795__3488(struct mem_Allocator__206 const a0) {
 struct mem_Allocator__206 const *t1;
 struct mem_Allocator__206 t2;
 struct mem_Allocator__206 t0;
 uintptr_t t3;
 nav__3488_51 t4;
 nav__3488_40 t6;
 uint8_t *t7;
 struct instruments_root_InstrumentHooks__188 *t8;
 uint16_t t5;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = (*t1);
 t3 = (uintptr_t)zig_return_address();
 t4 = mem_Allocator_allocBytesWithAlignment__anon_1889__3737(t2, (uintptr_t)72ul, t3);
 if (t4.error) {
  t5 = t4.error;
  t6.payload = ((struct instruments_root_InstrumentHooks__188 *)(uintptr_t)0xaaaaaaaaaaaaaaaaul);
  t6.error = t5;
  return t6;
 }
 t7 = t4.payload;
 t8 = (struct instruments_root_InstrumentHooks__188 *)t7;
 t6.payload = t8;
 t6.error = UINT16_C(0);
 return t6;
}

static struct instruments_valgrind_ValgrindInstrument__195 instruments_valgrind_ValgrindInstrument_init__325(struct mem_Allocator__206 const a0) {
 struct instruments_valgrind_ValgrindInstrument__195 t0;
 t0.allocator = a0;
 return t0;
}

static uint16_t fs_accessAbsolute__1053(nav__1053_39 const a0, struct fs_File_OpenFlags__1807 const a1) {
 struct fs_Dir__1809 const *t3;
 struct fs_Dir__1809 t1;
 struct fs_Dir__1809 t2;
 uint16_t t4;
 bool t0;
 t0 = fs_path_isAbsolute__3772(a0);
 debug_assert__179(t0);
 t1 = fs_cwd__1045();
 t2 = t1;
 t3 = (struct fs_Dir__1809 const *)&t2;
 t1 = (*t3);
 t4 = fs_Dir_access__3693(t1, a0, a1);
 if (t4) {
  return t4;
 }
 return 0;
}

static nav__1050_39 fs_openFileAbsolute__1050(nav__1050_41 const a0, struct fs_File_OpenFlags__1807 const a1) {
 struct fs_Dir__1809 const *t3;
 struct fs_Dir__1809 t1;
 struct fs_Dir__1809 t2;
 nav__1050_39 t4;
 bool t0;
 t0 = fs_path_isAbsolute__3772(a0);
 debug_assert__179(t0);
 t1 = fs_cwd__1045();
 t2 = t1;
 t3 = (struct fs_Dir__1809 const *)&t2;
 t1 = (*t3);
 t4 = fs_Dir_openFile__3629(t1, a0, a1);
 return t4;
}

static struct fifo_UnixPipe_Writer__538 fifo_UnixPipe_Writer_init__988(struct fs_File__546 const a0, struct mem_Allocator__206 const a1) {
 struct fs_File__546 *t1;
 struct mem_Allocator__206 *t2;
 struct fifo_UnixPipe_Writer__538 t0;
 t1 = (struct fs_File__546 *)&t0.file;
 (*t1) = a0;
 t2 = (struct mem_Allocator__206 *)&t0.allocator;
 (*t2) = a1;
 return t0;
}

static struct fifo_UnixPipe_Reader__540 fifo_UnixPipe_Reader_init__993(struct fs_File__546 const a0, struct mem_Allocator__206 const a1) {
 struct fs_File__546 *t1;
 struct mem_Allocator__206 *t2;
 struct fifo_UnixPipe_Reader__540 t0;
 t1 = (struct fs_File__546 *)&t0.file;
 (*t1) = a0;
 t2 = (struct mem_Allocator__206 *)&t0.allocator;
 (*t2) = a1;
 return t0;
}

static uint16_t fifo_UnixPipe_Writer_sendCmd__991(struct fifo_UnixPipe_Writer__538 *const a0, struct shared_Command__1919 const a1) {
 struct fifo_UnixPipe_Writer__538 *const *t1;
 struct mem_Allocator__206 *t3;
 struct mem_Allocator__206 t4;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t5;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t2;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t6;
 nav__991_55 t8;
 struct fifo_UnixPipe_Writer__538 *t9;
 struct fifo_UnixPipe_Writer__538 *t0;
 struct fs_File__546 *t10;
 uintptr_t t12;
 uint32_t const *t15;
 uint8_t const (*t16)[4];
 nav__991_44 t17;
 struct fs_File__546 t11;
 uint32_t t13;
 uint32_t t14;
 uint16_t t7;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Writer__538 *const *)&t0;
 t3 = (struct mem_Allocator__206 *)&a0->allocator;
 t4 = (*t3);
 t5 = array_list_ArrayListAligned_28u8_2cnull_29_init__3885(t4);
 t2 = t5;
 t6 = array_list_ArrayListAligned_28u8_2cnull_29_writer__3910(&t2);
 t7 = bincode_serialize__anon_1991__4058(t6, a1);
 if (t7) {
  t5 = t2;
  array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(t5);
  return t7;
 }
 t5 = t2;
 t8 = t5.items;
 t9 = (*t1);
 t10 = (struct fs_File__546 *)&t9->file;
 t11 = (*t10);
 t12 = t8.len;
 t13 = (uint32_t)t12;
 t14 = t13;
 t15 = (uint32_t const *)&t14;
 t16 = mem_asBytes__anon_2032__4059(t15);
 t17.ptr = &(*t16)[(uintptr_t)0ul];
 t17.len = (uintptr_t)4ul;
 t7 = fs_File_writeAll__1144(t11, t17);
 if (t7) {
  t5 = t2;
  array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(t5);
  return t7;
 }
 t9 = (*t1);
 t10 = (struct fs_File__546 *)&t9->file;
 t11 = (*t10);
 memcpy(&t17, &t8, sizeof(nav__991_44));
 t7 = fs_File_writeAll__1144(t11, t17);
 if (t7) {
  t5 = t2;
  array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(t5);
  return t7;
 }
 t5 = t2;
 array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(t5);
 return 0;
}

static nav__4169_38 bincode_deserializeInt__anon_2267__4169(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0) {
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t1;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t4;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t2;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t0;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t3;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *const *t5;
 void const **t7;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t8;
 void const *t9;
 nav__4169_50 (**t10)(void const *, nav__4169_52);
 struct io_Reader__2282 t11;
 struct io_Reader__2282 t6;
 struct io_Reader__2282 t12;
 struct io_Reader__2282 t15;
 struct io_Reader__2282 const *t13;
 uint8_t const (*t21)[4];
 nav__4169_38 t14;
 nav__4169_38 t18;
 uint32_t t22;
 nav__4169_64 t16;
 uint16_t t17;
 uint8_t t19[4];
 uint8_t t20[4];
 t0 = a0;
 t1 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *)&t0;
 t2 = (*t1);
 t3 = t2;
 t1 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *)&t3;
 t4 = t1;
 t5 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *const *)&t4;
 t7 = (void const **)&t6.context;
 t1 = (*t5);
 t8 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t1->context;
 t9 = (void const *)t8;
 (*t7) = t9;
 t10 = (nav__4169_50 (**)(void const *, nav__4169_52))&t6.readFn;
 (*t10) = &io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29_typeEras__4165;
 t11 = t6;
 t12 = t11;
 t13 = (struct io_Reader__2282 const *)&t12;
 t11 = (*t13);
 t15 = t11;
 t13 = (struct io_Reader__2282 const *)&t15;
 t11 = (*t13);
 t16 = io_Reader_readBytesNoEof__anon_2313__4207(t11);
 if (t16.error) {
  t17 = t16.error;
  t18.payload = UINT32_C(0xaaaaaaaa);
  t18.error = t17;
  t14 = t18;
  goto zig_block_0;
 }
 memcpy(t19, t16.payload, sizeof(uint8_t[4]));
 memcpy((char *)&t20, t19, sizeof(uint8_t[4]));
 t21 = (uint8_t const (*)[4])&t20;
 memcpy(t19, (const char *)t21, sizeof(uint8_t[4]));
 memcpy(&t22, &t19, sizeof(uint32_t));
 t22 = zig_wrap_u32(t22, UINT8_C(32));
 t18.payload = t22;
 t18.error = UINT16_C(0);
 t14 = t18;
 goto zig_block_0;

 zig_block_0:;
 memcpy(&t18, &t14, sizeof(nav__4169_38));
 if (t18.error) {
  t17 = t18.error;
  t18.payload = UINT32_C(0xaaaaaaaa);
  t18.error = t17;
  return t18;
 }
 t22 = t18.payload;
 t18.payload = t22;
 t18.error = UINT16_C(0);
 return t18;
}

static nav__4168_38 bincode_deserializeAlloc__anon_2256__4168(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4168_38 t0;
 uint32_t t2;
 uint16_t t1;
 (void)a1;
 t0 = bincode_deserializeInt__anon_2267__4169(a0);
 if (t0.error) {
  t1 = t0.error;
  t0.payload = UINT32_C(0xaaaaaaaa);
  t0.error = t1;
  return t0;
 }
 t2 = t0.payload;
 t0.payload = t2;
 t0.error = UINT16_C(0);
 return t0;
}

static nav__4211_40 bincode_deserializePointerAlloc__anon_2393__4211(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t1;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t6;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t7;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *t32;
 struct mem_Allocator__206 const *t3;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t4;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t0;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t5;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t31;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *const *t8;
 void const **t10;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t11;
 void const *t12;
 nav__4211_59 (**t13)(void const *, nav__4211_39);
 struct io_Reader__2282 t14;
 struct io_Reader__2282 t9;
 struct io_Reader__2282 t15;
 struct io_Reader__2282 t18;
 struct io_Reader__2282 t33;
 struct io_Reader__2282 t34;
 struct io_Reader__2282 const *t16;
 nav__4211_69 t17;
 nav__4211_69 t21;
 uint8_t const (*t24)[8];
 uint64_t t25;
 uint64_t t38;
 nav__4211_40 t26;
 uintptr_t t27;
 uintptr_t t37;
 struct mem_Allocator__206 t29;
 struct mem_Allocator__206 t2;
 nav__4211_39 t30;
 nav__4211_39 t28;
 nav__4211_59 t35;
 nav__4211_59 t36;
 nav__4211_72 t19;
 uint16_t t20;
 uint8_t t22[8];
 uint8_t t23[8];
 bool t39;
 t0 = a0;
 t1 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *)&t0;
 t2 = a1;
 t3 = (struct mem_Allocator__206 const *)&t2;
 t4 = (*t1);
 t5 = t4;
 t6 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *)&t5;
 t7 = t6;
 t8 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *const *)&t7;
 t10 = (void const **)&t9.context;
 t6 = (*t8);
 t11 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t6->context;
 t12 = (void const *)t11;
 (*t10) = t12;
 t13 = (nav__4211_59 (**)(void const *, nav__4211_39))&t9.readFn;
 (*t13) = &io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29_typeEras__4165;
 t14 = t9;
 t15 = t14;
 t16 = (struct io_Reader__2282 const *)&t15;
 t14 = (*t16);
 t18 = t14;
 t16 = (struct io_Reader__2282 const *)&t18;
 t14 = (*t16);
 t19 = io_Reader_readBytesNoEof__anon_2406__4212(t14);
 if (t19.error) {
  t20 = t19.error;
  t21.payload = UINT64_C(0xaaaaaaaaaaaaaaaa);
  t21.error = t20;
  t17 = t21;
  goto zig_block_0;
 }
 memcpy(t22, t19.payload, sizeof(uint8_t[8]));
 memcpy((char *)&t23, t22, sizeof(uint8_t[8]));
 t24 = (uint8_t const (*)[8])&t23;
 memcpy(t22, (const char *)t24, sizeof(uint8_t[8]));
 memcpy(&t25, &t22, sizeof(uint64_t));
 t25 = zig_wrap_u64(t25, UINT8_C(64));
 t21.payload = t25;
 t21.error = UINT16_C(0);
 t17 = t21;
 goto zig_block_0;

 zig_block_0:;
 memcpy(&t21, &t17, sizeof(nav__4211_69));
 if (t21.error) {
  t20 = t21.error;
  t26.payload = (nav__4211_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t26.error = t20;
  return t26;
 }
 t25 = t21.payload;
 t27 = t25;
 t29 = (*t3);
 t26 = mem_Allocator_alloc__anon_2114__4100(t29, t27);
 if (t26.error) {
  t20 = t26.error;
  t26.payload = (nav__4211_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t26.error = t20;
  return t26;
 }
 t30 = t26.payload;
 t28 = t30;
 t4 = (*t1);
 t30 = t28;
 t31 = t4;
 t1 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *)&t31;
 t32 = t1;
 t8 = (struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const *const *)&t32;
 t10 = (void const **)&t33.context;
 t1 = (*t8);
 t11 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t1->context;
 t12 = (void const *)t11;
 (*t10) = t12;
 t13 = (nav__4211_59 (**)(void const *, nav__4211_39))&t33.readFn;
 (*t13) = &io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29_typeEras__4165;
 t14 = t33;
 t34 = t14;
 t16 = (struct io_Reader__2282 const *)&t34;
 t14 = (*t16);
 t35 = io_Reader_readAll__4172(t14, t30);
 memcpy(&t36, &t35, sizeof(nav__4211_59));
 t37 = t36.payload;
 t25 = t37;
 t38 = t27;
 t39 = t25 != t38;
 if (t39) {
  bincode_invalidProtocol__anon_2430__4213();
  zig_unreachable();
 }
 goto zig_block_1;

 zig_block_1:;
 t30 = t28;
 t26.payload = t30;
 t26.error = UINT16_C(0);
 return t26;
}

static nav__4210_40 bincode_deserializeAlloc__anon_2380__4210(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4210_57 t0;
 nav__4210_40 t2;
 nav__4210_55 t3;
 nav__4210_39 t4;
 uint16_t t1;
 t0 = bincode_deserializePointerAlloc__anon_2393__4211(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t2.payload = (nav__4210_39){(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t2.error = t1;
  return t2;
 }
 t3 = t0.payload;
 memcpy(&t4, &t3, sizeof(nav__4210_39));
 t2.payload = t4;
 t2.error = UINT16_C(0);
 return t2;
}

static nav__4209_39 bincode_deserializeStructAlloc__anon_2355__4209(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 uint32_t *t1;
 nav__4209_39 t4;
 nav__4209_44 *t6;
 nav__4209_60 t7;
 nav__4209_44 t8;
 struct shared_Command__struct_1922__1922 t9;
 struct shared_Command__struct_1922__1922 t0;
 nav__4209_57 t2;
 uint32_t t5;
 uint16_t t3;
 t1 = (uint32_t *)&t0.pid;
 t2 = bincode_deserializeAlloc__anon_2256__4168(a0, a1);
 if (t2.error) {
  t3 = t2.error;
  t4.payload = (struct shared_Command__struct_1922__1922){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)};
  t4.error = t3;
  return t4;
 }
 t5 = t2.payload;
 (*t1) = t5;
 t6 = (nav__4209_44 *)&t0.uri;
 t7 = bincode_deserializeAlloc__anon_2380__4210(a0, a1);
 if (t7.error) {
  t3 = t7.error;
  t4.payload = (struct shared_Command__struct_1922__1922){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)};
  t4.error = t3;
  return t4;
 }
 t8 = t7.payload;
 (*t6) = t8;
 t9 = t0;
 t4.payload = t9;
 t4.error = UINT16_C(0);
 return t4;
}

static nav__4208_39 bincode_deserializeAlloc__anon_2326__4208(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4208_39 t0;
 struct shared_Command__struct_1922__1922 t2;
 uint16_t t1;
 t0 = bincode_deserializeStructAlloc__anon_2355__4209(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t0.payload = (struct shared_Command__struct_1922__1922){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)};
  t0.error = t1;
  return t0;
 }
 t2 = t0.payload;
 t0.payload = t2;
 t0.error = UINT16_C(0);
 return t0;
}

static uint16_t bincode_deserializeAlloc__anon_2445__4214(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 (void)a0;
 (void)a1;
 return 0;
}

static nav__4216_39 bincode_deserializeStructAlloc__anon_2507__4216(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4216_44 *t1;
 nav__4216_57 t2;
 nav__4216_39 t4;
 nav__4216_44 t5;
 struct shared_Command__struct_1923__1923 t6;
 struct shared_Command__struct_1923__1923 t0;
 uint16_t t3;
 t1 = (nav__4216_44 *)&t0.name;
 t2 = bincode_deserializeAlloc__anon_2380__4210(a0, a1);
 if (t2.error) {
  t3 = t2.error;
  t4.payload = (struct shared_Command__struct_1923__1923){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul}};
  t4.error = t3;
  return t4;
 }
 t5 = t2.payload;
 (*t1) = t5;
 t1 = (nav__4216_44 *)&t0.version;
 t2 = bincode_deserializeAlloc__anon_2380__4210(a0, a1);
 if (t2.error) {
  t3 = t2.error;
  t4.payload = (struct shared_Command__struct_1923__1923){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul}};
  t4.error = t3;
  return t4;
 }
 t5 = t2.payload;
 (*t1) = t5;
 t6 = t0;
 t4.payload = t6;
 t4.error = UINT16_C(0);
 return t4;
}

static nav__4215_39 bincode_deserializeAlloc__anon_2488__4215(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4215_39 t0;
 struct shared_Command__struct_1923__1923 t2;
 uint16_t t1;
 t0 = bincode_deserializeStructAlloc__anon_2507__4216(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t0.payload = (struct shared_Command__struct_1923__1923){{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul}};
  t0.error = t1;
  return t0;
 }
 t2 = t0.payload;
 t0.payload = t2;
 t0.error = UINT16_C(0);
 return t0;
}

static nav__4167_39 bincode_deserializeUnionAlloc__anon_2251__4167(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4167_39 t2;
 nav__4167_63 t6;
 struct shared_Command__struct_1922__1922 t7;
 struct shared_Command__1919 t8;
 nav__4167_65 t9;
 struct shared_Command__struct_1923__1923 t10;
 nav__4167_61 t0;
 uint32_t t3;
 uint16_t t1;
 uint8_t t4;
 bool t5;
 t0 = bincode_deserializeAlloc__anon_2256__4168(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t2.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
  t2.error = t1;
  return t2;
 }
 t3 = t0.payload;
 t4 = (uint8_t)t3;
 t5 = t4 == UINT8_C(0);
 if (t5) {
  t6 = bincode_deserializeAlloc__anon_2326__4208(a0, a1);
  if (t6.error) {
   t1 = t6.error;
   t2.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
   t2.error = t1;
   return t2;
  }
  t7 = t6.payload;
  t8.tag = UINT8_C(0);
  t8.payload.ExecutedBenchmark = t7;
  t2.payload = t8;
  t2.error = UINT16_C(0);
  return t2;
 }
 goto zig_block_0;

 zig_block_0:;
 t5 = t4 == UINT8_C(1);
 if (t5) {
  (void)bincode_deserializeAlloc__anon_2445__4214(a0, a1);
  return (nav__4167_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(1)},0};
 }
 goto zig_block_1;

 zig_block_1:;
 t5 = t4 == UINT8_C(2);
 if (t5) {
  (void)bincode_deserializeAlloc__anon_2445__4214(a0, a1);
  return (nav__4167_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(2)},0};
 }
 goto zig_block_2;

 zig_block_2:;
 t5 = t4 == UINT8_C(3);
 if (t5) {
  (void)bincode_deserializeAlloc__anon_2445__4214(a0, a1);
  return (nav__4167_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(3)},0};
 }
 goto zig_block_3;

 zig_block_3:;
 t5 = t4 == UINT8_C(4);
 if (t5) {
  (void)bincode_deserializeAlloc__anon_2445__4214(a0, a1);
  return (nav__4167_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(4)},0};
 }
 goto zig_block_4;

 zig_block_4:;
 t5 = t4 == UINT8_C(5);
 if (t5) {
  t9 = bincode_deserializeAlloc__anon_2488__4215(a0, a1);
  if (t9.error) {
   t1 = t9.error;
   t2.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
   t2.error = t1;
   return t2;
  }
  t10 = t9.payload;
  t8.tag = UINT8_C(5);
  t8.payload.SetIntegration = t10;
  t2.payload = t8;
  t2.error = UINT16_C(0);
  return t2;
 }
 goto zig_block_5;

 zig_block_5:;
 t5 = t4 == UINT8_C(6);
 if (t5) {
  (void)bincode_deserializeAlloc__anon_2445__4214(a0, a1);
  return (nav__4167_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(6)},0};
 }
 goto zig_block_6;

 zig_block_6:;
 zig_unreachable();
}

static nav__4166_39 bincode_deserializeAlloc__anon_2164__4166(struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 const a0, struct mem_Allocator__206 const a1) {
 nav__4166_39 t0;
 struct shared_Command__1919 t2;
 uint16_t t1;
 t0 = bincode_deserializeUnionAlloc__anon_2251__4167(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t0.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
  t0.error = t1;
  return t0;
 }
 t2 = t0.payload;
 t0.payload = t2;
 t0.error = UINT16_C(0);
 return t0;
}

static nav__996_39 fifo_UnixPipe_Reader_recvCmd__996(struct fifo_UnixPipe_Reader__540 *const a0) {
 struct fifo_UnixPipe_Reader__540 *const *t1;
 struct fifo_UnixPipe_Reader__540 *t3;
 struct fifo_UnixPipe_Reader__540 *t0;
 struct fs_File__546 *t4;
 nav__996_66 t6;
 nav__996_69 t7;
 nav__996_39 t9;
 uint8_t const (*t10)[4];
 struct mem_Allocator__206 *t13;
 struct mem_Allocator__206 t14;
 uintptr_t t15;
 nav__996_73 t16;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t19;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t18;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t20;
 struct shared_Command__1919 t21;
 struct fs_File__546 t5;
 uint32_t t12;
 uint16_t t8;
 uint8_t t11[4];
 uint8_t t2[4];
 bool t17;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Reader__540 *const *)&t0;
 t3 = (*t1);
 t4 = (struct fs_File__546 *)&t3->file;
 t5 = (*t4);
 t6.ptr = &t2[(uintptr_t)0ul];
 t6.len = (uintptr_t)4ul;
 t7 = fs_File_readAll__1134(t5, t6);
 if (t7.error) {
  t8 = t7.error;
  t9.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
  t9.error = t8;
  return t9;
 }
 t10 = (uint8_t const (*)[4])&t2;
 memcpy(t11, (const char *)t10, sizeof(uint8_t[4]));
 memcpy(&t12, &t11, sizeof(uint32_t));
 t12 = zig_wrap_u32(t12, UINT8_C(32));
 t3 = (*t1);
 t13 = (struct mem_Allocator__206 *)&t3->allocator;
 t14 = (*t13);
 t15 = (uintptr_t)t12;
 t16 = mem_Allocator_alloc__anon_2114__4100(t14, t15);
 if (t16.error) {
  t8 = t16.error;
  t9.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
  t9.error = t8;
  return t9;
 }
 t6 = t16.payload;
 zig_loop_36:
 t3 = (*t1);
 t4 = (struct fs_File__546 *)&t3->file;
 t5 = (*t4);
 t7 = fs_File_readAll__1134(t5, t6);
 t17 = t7.error == UINT16_C(0);
 if (t17) {
  goto zig_block_2;
 }
 goto zig_block_1;

 zig_block_2:;
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_36;

 zig_block_0:;
 t19 = io_fixed_buffer_stream_fixedBufferStream__anon_2136__4132(t6);
 t18 = t19;
 t20 = io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_reader__4121(&t18);
 t13 = (struct mem_Allocator__206 *)&a0->allocator;
 t14 = (*t13);
 t9 = bincode_deserializeAlloc__anon_2164__4166(t20, t14);
 if (t9.error) {
  t8 = t9.error;
  t3 = (*t1);
  t13 = (struct mem_Allocator__206 *)&t3->allocator;
  t14 = (*t13);
  mem_Allocator_free__anon_2541__4217(t14, t6);
  t9.payload = (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)};
  t9.error = t8;
  return t9;
 }
 t21 = t9.payload;
 t3 = (*t1);
 t13 = (struct mem_Allocator__206 *)&t3->allocator;
 t14 = (*t13);
 mem_Allocator_free__anon_2541__4217(t14, t6);
 t9.payload = t21;
 t9.error = UINT16_C(0);
 return t9;
}

static nav__997_39 fifo_UnixPipe_Reader_waitForResponse__997(struct fifo_UnixPipe_Reader__540 *const a0, nav__997_42 const a1) {
 zig_i128 t2;
 zig_i128 t6;
 struct fifo_UnixPipe_Reader__540 *const *t1;
 uint64_t t3;
 uint64_t t5;
 struct shared_Command__1919 t7;
 struct shared_Command__1919 t10;
 struct fifo_UnixPipe_Reader__540 *t8;
 struct fifo_UnixPipe_Reader__540 *t0;
 nav__997_39 t9;
 bool t4;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Reader__540 *const *)&t0;
 t2 = time_nanoTimestamp__4072();
 t4 = a1.is_null != true;
 if (t4) {
  t5 = a1.payload;
  t3 = t5;
  goto zig_block_0;
 }
 t3 = UINT64_C(5000000000);
 goto zig_block_0;

 zig_block_0:;
 zig_loop_13:
 t6 = time_nanoTimestamp__4072();
 t6 = zig_sub_i128(t6, t2);
 t5 = zig_lo_i128(t6);
 t4 = t5 > t3;
 if (t4) {
  return (nav__997_39){{{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(0x2)},zig_error_AckTimeout};
 }
 goto zig_block_2;

 zig_block_2:;
 t8 = (*t1);
 t9 = fifo_UnixPipe_Reader_recvCmd__996(t8);
 t4 = t9.error == UINT16_C(0);
 if (t4) {
  t10 = t9.payload;
  t7 = t10;
  goto zig_block_3;
 }
 utils_sleep__4222(UINT64_C(10000000));
 goto zig_block_1;

 zig_block_3:;
 t9.payload = t7;
 t9.error = UINT16_C(0);
 return t9;

 zig_block_1:;
 goto zig_loop_13;
}

static uint16_t fifo_UnixPipe_Reader_waitForAck__998(struct fifo_UnixPipe_Reader__540 *const a0, nav__998_40 const a1) {
 struct fifo_UnixPipe_Reader__540 *const *t1;
 struct fifo_UnixPipe_Reader__540 *t2;
 struct fifo_UnixPipe_Reader__540 *t0;
 nav__998_55 t3;
 struct shared_Command__1919 t5;
 struct shared_Command__1919 t6;
 struct shared_Command__1919 const *t7;
 struct mem_Allocator__206 *t10;
 struct mem_Allocator__206 t11;
 uint16_t t4;
 uint8_t t8;
 bool t9;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Reader__540 *const *)&t0;
 t2 = (*t1);
 t3 = fifo_UnixPipe_Reader_waitForResponse__997(t2, a1);
 if (t3.error) {
  t4 = t3.error;
  return t4;
 }
 t5 = t3.payload;
 t6 = t5;
 t7 = (struct shared_Command__1919 const *)&t6;
 t8 = t5.tag;
 t9 = t8 == UINT8_C(3);
 if (t9) {
  t5 = (*t7);
  t10 = (struct mem_Allocator__206 *)&a0->allocator;
  t11 = (*t10);
  shared_Command_deinit__3835(t5, t11);
  return 0;
 }
 t8 = t5.tag;
 t9 = t8 == UINT8_C(6);
 if (t9) {
  t5 = (*t7);
  t10 = (struct mem_Allocator__206 *)&a0->allocator;
  t11 = (*t10);
  shared_Command_deinit__3835(t5, t11);
  return zig_error_UnexpectedError;
 }
 t5 = (*t7);
 t10 = (struct mem_Allocator__206 *)&a0->allocator;
 t11 = (*t10);
 shared_Command_deinit__3835(t5, t11);
 return zig_error_UnexpectedResponse;
}

static uint16_t instruments_perf_PerfInstrument_send_cmd__339(struct instruments_perf_PerfInstrument__200 *const a0, struct shared_Command__1919 const a1) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 struct fifo_UnixPipe_Reader__540 *t5;
 uint16_t t4;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 t4 = fifo_UnixPipe_Writer_sendCmd__991(t3, a1);
 if (t4) {
  return t4;
 }
 t2 = (*t1);
 t5 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 t4 = fifo_UnixPipe_Reader_waitForAck__998(t5, (nav__339_67){UINT64_C(0xaaaaaaaaaaaaaaaa),true});
 if (t4) {
  return t4;
 }
 return 0;
}

static bool instruments_perf_PerfInstrument_is_instrumented__340(struct instruments_perf_PerfInstrument__200 *const a0) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 uint16_t t3;
 bool t4;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = instruments_perf_PerfInstrument_send_cmd__339(t2, (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(4)});
 t4 = t3 == UINT16_C(0);
 if (t4) {
  goto zig_block_0;
 }
 return false;

 zig_block_0:;
 return true;
}

static nav__3737_39 mem_Allocator_allocBytesWithAlignment__anon_1889__3737(struct mem_Allocator__206 const a0, uintptr_t const a1, uintptr_t const a2) {
 struct mem_Allocator__206 const *t1;
 uint64_t t2;
 uint8_t *t4;
 uint8_t *t13;
 uint8_t *t14;
 uint8_t *t15;
 struct mem_Allocator__206 t5;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t7;
 struct mem_Allocator_VTable__209 const *const *t8;
 struct mem_Allocator_VTable__209 const *t9;
 uint8_t *(*const *t10)(void *, uintptr_t, uint8_t, uintptr_t);
 uint8_t *(*t11)(void *, uintptr_t, uint8_t, uintptr_t);
 void *t12;
 uint8_t *const *t16;
 nav__3737_52 t17;
 nav__3737_39 t18;
 bool t3;
 uint8_t t6;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = a1;
 t3 = t2 == UINT64_C(0);
 if (t3) {
  return (nav__3737_39){(uint8_t *)(uintptr_t)0xfffffffffffffff8ul,0};
 }
 goto zig_block_0;

 zig_block_0:;
 t5 = (*t1);
 t6 = mem_Alignment_fromByteUnits__628((uintptr_t)8ul);
 t7 = t5;
 t1 = (struct mem_Allocator__206 const *)&t7;
 t8 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t9 = (*t8);
 t10 = (uint8_t *(*const *)(void *, uintptr_t, uint8_t, uintptr_t))&t9->alloc;
 t11 = (*t10);
 t12 = t5.ptr;
 t13 = t11(t12, a1, t6, a2);
 t3 = t13 != NULL;
 if (t3) {
  t14 = t13;
  t4 = t14;
  goto zig_block_1;
 }
 return (nav__3737_39){((uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),zig_error_OutOfMemory};

 zig_block_1:;
 t15 = t4;
 t16 = (uint8_t *const *)&t15;
 t14 = (*t16);
 t14 = (uint8_t *)(((uintptr_t)t14) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t17.ptr = t14;
 t17.len = a1;
 t14 = (uint8_t *)t4;
 t18.payload = t14;
 t18.error = UINT16_C(0);
 return t18;
}

static bool fs_path_isAbsolute__3772(nav__3772_39 const a0) {
 bool t0;
 t0 = fs_path_isAbsolutePosix__3778(a0);
 return t0;
}

static void debug_assert__179(bool const a0) {
 bool t0;
 t0 = !a0;
 if (t0) {
  zig_unreachable();
 }
 goto zig_block_0;

 zig_block_0:;
 return;
}

static struct fs_Dir__1809 fs_cwd__1045(void) {
 struct fs_Dir__1809 t0;
 t0 = (struct fs_Dir__1809){-INT32_C(100)};
 return t0;
}

static uint16_t fs_Dir_access__3693(struct fs_Dir__1809 const a0, nav__3693_40 const a1, struct fs_File_OpenFlags__1807 const a2) {
 struct fs_Dir__1809 const *t1;
 uint8_t const (*t6)[4096];
 uint8_t const *t8;
 struct fs_Dir__1809 t7;
 struct fs_Dir__1809 t0;
 nav__3693_49 t2;
 uint16_t t3;
 uint8_t t4[4096];
 uint8_t t5[4096];
 t0 = a0;
 t1 = (struct fs_Dir__1809 const *)&t0;
 t2 = posix_toPosixPath__1628(a1);
 if (t2.error) {
  t3 = t2.error;
  return t3;
 }
 memcpy(t4, t2.payload, sizeof(uint8_t[4096]));
 memcpy((char *)&t5, t4, sizeof(uint8_t[4096]));
 t6 = (uint8_t const (*)[4096])&t5;
 t7 = (*t1);
 t8 = (uint8_t const *)t6;
 t3 = fs_Dir_accessZ__3694(t7, t8, a2);
 return t3;
}

static nav__3629_39 fs_Dir_openFile__3629(struct fs_Dir__1809 const a0, nav__3629_42 const a1, struct fs_File_OpenFlags__1807 const a2) {
 struct fs_Dir__1809 const *t1;
 uint8_t const (*t8)[4096];
 uint8_t const *t10;
 nav__3629_39 t5;
 struct fs_Dir__1809 t9;
 struct fs_Dir__1809 t0;
 nav__3629_53 t3;
 uint16_t t4;
 uint8_t t6[4096];
 uint8_t t7[4096];
 struct fs_File_OpenFlags__1807 t2;
 t0 = a0;
 t1 = (struct fs_Dir__1809 const *)&t0;
 t2 = a2;
 t3 = posix_toPosixPath__1628(a1);
 if (t3.error) {
  t4 = t3.error;
  t5.payload = (struct fs_File__546){-INT32_C(0x55555556)};
  t5.error = t4;
  return t5;
 }
 memcpy(t6, t3.payload, sizeof(uint8_t[4096]));
 memcpy((char *)&t7, t6, sizeof(uint8_t[4096]));
 t8 = (uint8_t const (*)[4096])&t7;
 t9 = (*t1);
 t10 = (uint8_t const *)t8;
 t5 = fs_Dir_openFileZ__3630(t9, t10, a2);
 return t5;
}

static struct array_list_ArrayListAligned_28u8_2cnull_29__1950 array_list_ArrayListAligned_28u8_2cnull_29_init__3885(struct mem_Allocator__206 const a0) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t0;
 t0.items = (nav__3885_42){(uint8_t *)((void const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),(uintptr_t)0ul};
 t0.capacity = (uintptr_t)0ul;
 t0.allocator = a0;
 return t0;
}

static struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 array_list_ArrayListAligned_28u8_2cnull_29_writer__3910(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 **t1;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 **)&t0.context;
 (*t1) = a0;
 return t0;
}

static uint16_t bincode_serialize__anon_1991__4058(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__1919 const a1) {
 uint16_t t0;
 t0 = bincode_serializeUnion__anon_2692__4236(a0, a1);
 if (t0) {
  return t0;
 }
 return 0;
}

static void array_list_ArrayListAligned_28u8_2cnull_29_deinit__3887(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const a0) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const *t1;
 struct mem_Allocator__206 const *t2;
 struct mem_Allocator__206 t3;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t4;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t0;
 nav__3887_41 t5;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const *)&t0;
 t2 = (struct mem_Allocator__206 const *)&t1->allocator;
 t3 = (*t2);
 t4 = (*t1);
 t5 = array_list_ArrayListAligned_28u8_2cnull_29_allocatedSlice__3933(t4);
 mem_Allocator_free__anon_2541__4217(t3, t5);
 return;
}

static uint8_t const (*mem_asBytes__anon_2032__4059(uint32_t const *const a0))[4] {
 uint8_t const (*t0)[4];
 t0 = (uint8_t const (*)[4])a0;
 return t0;
}

static uint16_t fs_File_writeAll__1144(struct fs_File__546 const a0, nav__1144_40 const a1) {
 struct fs_File__546 const *t1;
 nav__1144_40 const *t3;
 uintptr_t t5;
 uintptr_t t6;
 uintptr_t t13;
 uintptr_t t4;
 uint64_t t7;
 uint64_t t8;
 nav__1144_40 t11;
 nav__1144_40 t2;
 uint8_t const *t12;
 nav__1144_47 t14;
 struct fs_File__546 t10;
 struct fs_File__546 t0;
 uint16_t t15;
 bool t9;
 t0 = a0;
 t1 = (struct fs_File__546 const *)&t0;
 t2 = a1;
 t3 = (nav__1144_40 const *)&t2;
 t4 = (uintptr_t)0ul;
 zig_loop_11:
 t5 = t4;
 t6 = a1.len;
 t7 = t5;
 t8 = t6;
 t9 = t7 < t8;
 if (t9) {
  t6 = t4;
  t10 = (*t1);
  t5 = t4;
  t11 = (*t3);
  t12 = t11.ptr;
  t12 = (uint8_t const *)(((uintptr_t)t12) + (t5*sizeof(uint8_t)));
  t13 = t11.len;
  t5 = t13 - t5;
  t11.ptr = t12;
  t11.len = t5;
  t14 = fs_File_write__1143(t10, t11);
  if (t14.error) {
   t15 = t14.error;
   return t15;
  }
  t5 = t14.payload;
  t5 = t6 + t5;
  t4 = t5;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_11;

 zig_block_0:;
 return 0;
}

static zig_i128 time_nanoTimestamp__4072(void) {
 zig_i128 t6;
 zig_i128 t7;
 nav__4072_41 t0;
 struct os_linux_timespec__struct_2728__2728 t2;
 struct os_linux_timespec__struct_2728__2728 t3;
 intptr_t t5;
 uint16_t t4;
 bool t1;
 t0 = posix_clock_gettime__1544(UINT32_C(0));
 t1 = t0.error == UINT16_C(0);
 if (t1) {
  t3 = t0.payload;
  t2 = t3;
  goto zig_block_0;
 }
 t4 = t0.error;
 switch (t4) {
  case zig_error_UnsupportedClock:
  case zig_error_Unexpected: {
   return zig_make_i128(INT64_C(0), UINT64_C(0));
  }
  default: zig_unreachable();
 }

 zig_block_0:;
 t5 = t2.sec;
 t6 = zig_make_i128(0, t5);
 t6 = zig_mul_i128(t6, zig_make_i128(INT64_C(0), UINT64_C(1000000000)));
 t5 = t2.nsec;
 t7 = zig_make_i128(0, t5);
 t7 = zig_add_i128(t6, t7);
 return t7;
}

static nav__1134_38 fs_File_readAll__1134(struct fs_File__546 const a0, nav__1134_41 const a1) {
 struct fs_File__546 const *t1;
 nav__1134_41 const *t3;
 uintptr_t t5;
 uintptr_t t6;
 uintptr_t t4;
 uint64_t t7;
 uint64_t t8;
 nav__1134_41 t11;
 nav__1134_41 t2;
 uint8_t *t12;
 nav__1134_38 t13;
 struct fs_File__546 t10;
 struct fs_File__546 t0;
 uint16_t t14;
 bool t9;
 t0 = a0;
 t1 = (struct fs_File__546 const *)&t0;
 t2 = a1;
 t3 = (nav__1134_41 const *)&t2;
 t4 = (uintptr_t)0ul;
 zig_loop_11:
 t5 = t4;
 t6 = a1.len;
 t7 = t5;
 t8 = t6;
 t9 = t7 != t8;
 if (t9) {
  t10 = (*t1);
  t6 = t4;
  t11 = (*t3);
  t12 = t11.ptr;
  t12 = (uint8_t *)(((uintptr_t)t12) + (t6*sizeof(uint8_t)));
  t5 = t11.len;
  t6 = t5 - t6;
  t11.ptr = t12;
  t11.len = t6;
  t13 = fs_File_read__1133(t10, t11);
  if (t13.error) {
   t14 = t13.error;
   t13.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
   t13.error = t14;
   return t13;
  }
  t6 = t13.payload;
  t8 = t6;
  t9 = t8 == UINT64_C(0);
  if (t9) {
   goto zig_block_0;
  }
  goto zig_block_2;

  zig_block_2:;
  t5 = t4;
  t6 = t5 + t6;
  t4 = t6;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_11;

 zig_block_0:;
 t6 = t4;
 t13.payload = t6;
 t13.error = UINT16_C(0);
 return t13;
}

static nav__4100_40 mem_Allocator_alloc__anon_2114__4100(struct mem_Allocator__206 const a0, uintptr_t const a1) {
 struct mem_Allocator__206 const *t1;
 struct mem_Allocator__206 t2;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t5;
 uintptr_t t3;
 nav__4100_40 t4;
 nav__4100_40 t8;
 nav__4100_51 t6;
 uint8_t *t9;
 uint8_t *t10;
 uint8_t *const *t11;
 nav__4100_39 t12;
 uint16_t t7;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = (*t1);
 t3 = (uintptr_t)zig_return_address();
 t5 = t2;
 t1 = (struct mem_Allocator__206 const *)&t5;
 t2 = (*t1);
 t6 = mem_Allocator_allocWithSizeAndAlignment__anon_2759__4237(t2, a1, t3);
 if (t6.error) {
  t7 = t6.error;
  t8.payload = (nav__4100_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t8.error = t7;
  t4 = t8;
  goto zig_block_0;
 }
 t9 = t6.payload;
 t10 = t9;
 t11 = (uint8_t *const *)&t10;
 t9 = (*t11);
 t9 = (uint8_t *)(((uintptr_t)t9) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t12.ptr = t9;
 t12.len = a1;
 t8.payload = t12;
 t8.error = UINT16_C(0);
 t4 = t8;
 goto zig_block_0;

 zig_block_0:;
 return t4;
}

static struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 io_fixed_buffer_stream_fixedBufferStream__anon_2136__4132(nav__4132_40 const a0) {
 nav__4132_40 *t1;
 uintptr_t *t2;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t0;
 t1 = (nav__4132_40 *)&t0.buffer;
 (*t1) = a0;
 t2 = (uintptr_t *)&t0.pos;
 (*t2) = (uintptr_t)0ul;
 return t0;
}

static struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_reader__4121(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const a0) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 **t1;
 struct io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29__2156 t0;
 t1 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 **)&t0.context;
 (*t1) = a0;
 return t0;
}

static nav__4165_38 io_GenericReader_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7b_7d_2c_28function_20_27read_27_29_29_typeEras__4165(void const *const a0, nav__4165_41 const a1) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t0;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t1;
 nav__4165_38 t2;
 nav__4165_38 t3;
 t0 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)a0;
 t1 = (*t0);
 t2 = io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_read__4124(t1, a1);
 memcpy(&t3, &t2, sizeof(nav__4165_38));
 return t3;
}

static nav__4207_39 io_Reader_readBytesNoEof__anon_2313__4207(struct io_Reader__2282 const a0) {
 struct io_Reader__2282 const *t1;
 struct io_Reader__2282 t3;
 struct io_Reader__2282 t0;
 nav__4207_46 t4;
 uint16_t t5;
 nav__4207_39 t6;
 uint8_t t7[4];
 uint8_t t2[4];
 t0 = a0;
 t1 = (struct io_Reader__2282 const *)&t0;
 t3 = (*t1);
 t4.ptr = &t2[(uintptr_t)0ul];
 t4.len = (uintptr_t)4ul;
 t5 = io_Reader_readNoEof__4174(t3, t4);
 if (t5) {
  memcpy(t6.payload, "\252\252\252\252", sizeof(uint8_t[4]));
  t6.error = t5;
  return t6;
 }
 memcpy(t7, (const char *)&t2, sizeof(uint8_t[4]));
 memcpy(t6.payload, t7, sizeof(uint8_t[4]));
 t6.error = UINT16_C(0);
 return t6;
}

static nav__4212_39 io_Reader_readBytesNoEof__anon_2406__4212(struct io_Reader__2282 const a0) {
 struct io_Reader__2282 const *t1;
 struct io_Reader__2282 t3;
 struct io_Reader__2282 t0;
 nav__4212_46 t4;
 uint16_t t5;
 nav__4212_39 t6;
 uint8_t t7[8];
 uint8_t t2[8];
 t0 = a0;
 t1 = (struct io_Reader__2282 const *)&t0;
 t3 = (*t1);
 t4.ptr = &t2[(uintptr_t)0ul];
 t4.len = (uintptr_t)8ul;
 t5 = io_Reader_readNoEof__4174(t3, t4);
 if (t5) {
  memcpy(t6.payload, "\252\252\252\252\252\252\252\252", sizeof(uint8_t[8]));
  t6.error = t5;
  return t6;
 }
 memcpy(t7, (const char *)&t2, sizeof(uint8_t[8]));
 memcpy(t6.payload, t7, sizeof(uint8_t[8]));
 t6.error = UINT16_C(0);
 return t6;
}

static nav__4172_38 io_Reader_readAll__4172(struct io_Reader__2282 const a0, nav__4172_41 const a1) {
 uintptr_t t0;
 nav__4172_38 t1;
 t0 = a1.len;
 t1 = io_Reader_readAtLeast__4173(a0, a1, t0);
 return t1;
}

static zig_cold zig_noreturn void bincode_invalidProtocol__anon_2430__4213(void) {
 debug_no_panic_call__4239((nav__4213_40){(uint8_t const *)&__anon_2779,(uintptr_t)88ul}, (nav__4213_42){(uintptr_t)0xaaaaaaaaaaaaaaaaul,true});
 zig_unreachable();
}

static void mem_Allocator_free__anon_2541__4217(struct mem_Allocator__206 const a0, nav__4217_40 const a1) {
 struct mem_Allocator__206 const *t1;
 nav__4217_40 t2;
 uintptr_t t3;
 uint64_t t4;
 uint8_t *t6;
 uint8_t *t7;
 uint8_t *t8;
 uint8_t *const *t9;
 struct mem_Allocator__206 t10;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t12;
 struct mem_Allocator_VTable__209 const *const *t13;
 struct mem_Allocator_VTable__209 const *t14;
 void (*const *t15)(void *, nav__4217_40, uint8_t, uintptr_t);
 void (*t16)(void *, nav__4217_40, uint8_t, uintptr_t);
 void *t17;
 bool t5;
 uint8_t t11;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = mem_sliceAsBytes__anon_2876__4266(a1);
 t3 = t2.len;
 t4 = t3;
 t5 = t4 == UINT64_C(0);
 if (t5) {
  return;
 }
 goto zig_block_0;

 zig_block_0:;
 t6 = t2.ptr;
 t7 = (uint8_t *)t6;
 t8 = t7;
 t9 = (uint8_t *const *)&t8;
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t2.ptr = t7;
 t2.len = t3;
 t10 = (*t1);
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t2.ptr = t7;
 t2.len = t3;
 t11 = mem_Alignment_fromByteUnits__628((uintptr_t)1ul);
 t3 = (uintptr_t)zig_return_address();
 t12 = t10;
 t1 = (struct mem_Allocator__206 const *)&t12;
 t13 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t14 = (*t13);
 t15 = (void (*const *)(void *, nav__4217_40, uint8_t, uintptr_t))&t14->free;
 t16 = (*t15);
 t17 = t10.ptr;
 t16(t17, t2, t11, t3);
 return;
}

static void utils_sleep__4222(uint64_t const a0) {
 uint64_t t0;
 uint64_t t1;
 long *t3;
 long t4;
 long t7;
 nav__4222_42 t5;
 struct cimport_struct_timespec__2902 const *t9;
 struct cimport_struct_timespec__2902 *t10;
 struct cimport_struct_timespec__2902 t13;
 struct cimport_struct_timespec__2902 t2;
 struct cimport_struct_timespec__2902 t8;
 int t11;
 int32_t t12;
 bool t6;
 t0 = a0 / UINT64_C(1000000000);
 t1 = a0 % UINT64_C(1000000000);
 t3 = (long *)&t2.tv_sec;
 t5 = math_cast__anon_2915__5112(t0);
 t6 = t5.is_null != true;
 if (t6) {
  t7 = t5.payload;
  t4 = t7;
  goto zig_block_0;
 }
 t4 = LONG_MAX;
 goto zig_block_0;

 zig_block_0:;
 (*t3) = t4;
 t3 = (long *)&t2.tv_nsec;
 t5 = math_cast__anon_2915__5112(t1);
 t6 = t5.is_null != true;
 if (t6) {
  t7 = t5.payload;
  t4 = t7;
  goto zig_block_1;
 }
 t4 = LONG_MAX;
 goto zig_block_1;

 zig_block_1:;
 (*t3) = t4;
 zig_loop_28:
 t9 = (struct cimport_struct_timespec__2902 const *)&t2;
 t10 = (struct cimport_struct_timespec__2902 *)&t8;
 t11 = nanosleep(t9, t10);
 t12 = t11;
 t6 = t12 == INT32_C(4);
 if (t6) {
  t13 = t8;
  t2 = t13;
  goto zig_block_2;
 }
 return;

 zig_block_2:;
 goto zig_loop_28;
}

static void shared_Command_deinit__3835(struct shared_Command__1919 const a0, struct mem_Allocator__206 const a1) {
 struct mem_Allocator__206 const *t1;
 struct shared_Command__struct_1923__1923 t3;
 struct mem_Allocator__206 t4;
 struct mem_Allocator__206 t0;
 nav__3835_43 t5;
 struct shared_Command__struct_1922__1922 t6;
 uint8_t t2;
 t0 = a1;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = a0.tag;
 switch (t2) {
  case UINT8_C(5): {
   t3 = a0.payload.SetIntegration;
   t4 = (*t1);
   t5 = t3.name;
   mem_Allocator_free__anon_2982__5883(t4, t5);
   t4 = (*t1);
   t5 = t3.version;
   mem_Allocator_free__anon_2982__5883(t4, t5);
   goto zig_block_0;
  }
  case UINT8_C(0): {
   t6 = a0.payload.ExecutedBenchmark;
   t4 = (*t1);
   t5 = t6.uri;
   mem_Allocator_free__anon_2982__5883(t4, t5);
   goto zig_block_0;
  }
  default: {
   goto zig_block_0;
  }
 }

 zig_block_0:;
 return;
}

static uint8_t mem_Alignment_fromByteUnits__628(uintptr_t const a0) {
 bool t0;
 uint8_t t1;
 t0 = math_isPowerOfTwo__anon_2984__5884(a0);
 debug_assert__179(t0);
 t1 = zig_ctz_u64(a0, UINT8_C(64));
 return t1;
}

static bool fs_path_isAbsolutePosix__3778(nav__3778_39 const a0) {
 uintptr_t t0;
 uint64_t t1;
 bool t2;
 bool t3;
 uint8_t t4;
 t0 = a0.len;
 t1 = t0;
 t2 = t1 > UINT64_C(0);
 if (t2) {
  t4 = a0.ptr[(uintptr_t)0ul];
  t2 = t4 == UINT8_C(47);
  t3 = t2;
  goto zig_block_0;
 }
 t3 = false;
 goto zig_block_0;

 zig_block_0:;
 return t3;
}

static nav__1628_39 posix_toPosixPath__1628(nav__1628_41 const a0) {
 uintptr_t t1;
 uint64_t t2;
 uint8_t *t4;
 nav__1628_47 t5;
 uint8_t const *t6;
 nav__1628_39 t8;
 bool t3;
 uint8_t t7[4096];
 uint8_t t0[4096];
 t1 = a0.len;
 t2 = t1;
 t3 = t2 >= UINT64_C(4096);
 if (t3) {
  return (nav__1628_39){zig_error_NameTooLong,{'\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa','\xaa'}};
 }
 goto zig_block_0;

 zig_block_0:;
 t1 = a0.len;
 t4 = (uint8_t *)&t0;
 t4 = (uint8_t *)(((uintptr_t)t4) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t5.ptr = t4;
 t5.len = t1;
 t6 = a0.ptr;
 if (t5.len != 0) memcpy(t5.ptr, t6, t5.len * sizeof(uint8_t));
 t1 = a0.len;
 t4 = (uint8_t *)&t0[t1];
 (*t4) = UINT8_C(0);
 memcpy(t7, (const char *)&t0, sizeof(uint8_t[4096]));
 memcpy(t8.payload, t7, sizeof(uint8_t[4096]));
 t8.error = UINT16_C(0);
 return t8;
}

static uint16_t fs_Dir_accessZ__3694(struct fs_Dir__1809 const a0, uint8_t const *const a1, struct fs_File_OpenFlags__1807 const a2) {
 uint32_t t2;
 int32_t t3;
 struct fs_Dir__1809 t0;
 uint16_t t4;
 uint8_t t1;
 t0 = a0;
 t1 = a2.mode;
 switch (t1) {
  case UINT8_C(0): {
   t2 = UINT32_C(0);
   goto zig_block_0;
  }
  case UINT8_C(1): {
   t2 = UINT32_C(2);
   goto zig_block_0;
  }
  case UINT8_C(2): {
   t2 = UINT32_C(6);
   goto zig_block_0;
  }
  default: zig_unreachable();
 }

 zig_block_0:;
 t3 = a0.fd;
 t4 = posix_faccessatZ__1519(t3, a1, t2, UINT32_C(0));
 return t4;
}

static nav__3630_39 fs_Dir_openFileZ__3630(struct fs_Dir__1809 const a0, uint8_t const *const a1, struct fs_File_OpenFlags__1807 const a2) {
 uint32_t *t2;
 struct fs_File__546 *t14;
 int32_t *t15;
 int32_t t6;
 int32_t t11;
 int32_t t12;
 uint32_t t7;
 uint32_t t1;
 nav__3630_50 t8;
 nav__3630_39 t10;
 nav__3630_39 t13;
 struct fs_Dir__1809 t0;
 uint16_t t9;
 uint8_t t3;
 uint8_t t4;
 bool t5;
 t0 = a0;
 t2 = (uint32_t *)&t1;
 t3 = a2.mode;
 switch (t3) {
  case UINT8_C(0): {
   t4 = UINT8_C(0);
   goto zig_block_0;
  }
  case UINT8_C(1): {
   t4 = UINT8_C(1);
   goto zig_block_0;
  }
  case UINT8_C(2): {
   t4 = UINT8_C(2);
   goto zig_block_0;
  }
  default: zig_unreachable();
 }

 zig_block_0:;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffffffc)), zig_shl_u32((uint32_t)t4, UINT8_C(0)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffffc3)), zig_shl_u32((uint32_t)UINT8_C(0), UINT8_C(2)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffffbf)), zig_shl_u32((uint32_t)false, UINT8_C(6)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffff7f)), zig_shl_u32((uint32_t)false, UINT8_C(7)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffffeff)), zig_shl_u32((uint32_t)false, UINT8_C(8)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffffdff)), zig_shl_u32((uint32_t)false, UINT8_C(9)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffffbff)), zig_shl_u32((uint32_t)false, UINT8_C(10)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffff7ff)), zig_shl_u32((uint32_t)false, UINT8_C(11)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffefff)), zig_shl_u32((uint32_t)false, UINT8_C(12)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffdfff)), zig_shl_u32((uint32_t)false, UINT8_C(13)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffffbfff)), zig_shl_u32((uint32_t)false, UINT8_C(14)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffff7fff)), zig_shl_u32((uint32_t)UINT8_C(0), UINT8_C(15)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffeffff)), zig_shl_u32((uint32_t)false, UINT8_C(16)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffdffff)), zig_shl_u32((uint32_t)false, UINT8_C(17)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffbffff)), zig_shl_u32((uint32_t)false, UINT8_C(18)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfff7ffff)), zig_shl_u32((uint32_t)false, UINT8_C(19)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffefffff)), zig_shl_u32((uint32_t)false, UINT8_C(20)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffdfffff)), zig_shl_u32((uint32_t)false, UINT8_C(21)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xffbfffff)), zig_shl_u32((uint32_t)false, UINT8_C(22)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0x7fffff)), zig_shl_u32((uint32_t)UINT16_C(0), UINT8_C(23)));
 t2 = (uint32_t *)&t1;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfff7ffff)), zig_shl_u32((uint32_t)true, UINT8_C(19)));
 t2 = (uint32_t *)&t1;
 t5 = a2.allow_ctty;
 t5 = !t5;
 (*t2) = zig_or_u32(zig_and_u32((*t2), UINT32_C(0xfffffeff)), zig_shl_u32((uint32_t)t5, UINT8_C(8)));
 t6 = a0.fd;
 t7 = t1;
 t8 = posix_openatZ__1375(t6, a1, t7, (uintptr_t)0ul);
 if (t8.error) {
  t9 = t8.error;
  t10.payload = (struct fs_File__546){-INT32_C(0x55555556)};
  t10.error = t9;
  return t10;
 }
 t6 = t8.payload;
 t4 = a2.lock;
 t5 = t4 != UINT8_C(0);
 if (t5) {
  t5 = a2.lock_nonblocking;
  if (t5) {
   t11 = INT32_C(4);
   goto zig_block_2;
  }
  t11 = INT32_C(0);
  goto zig_block_2;

  zig_block_2:;
  t4 = a2.lock;
  switch (t4) {
   case UINT8_C(0): {
    zig_unreachable();
   }
   case UINT8_C(1): {
    t11 = INT32_C(1) | t11;
    t12 = t11;
    goto zig_block_3;
   }
   case UINT8_C(2): {
    t11 = INT32_C(2) | t11;
    t12 = t11;
    goto zig_block_3;
   }
   default: zig_unreachable();
  }

  zig_block_3:;
  t9 = posix_flock__1536(t6, t12);
  if (t9) {
   posix_close__1333(t6);
   t10.payload = (struct fs_File__546){-INT32_C(0x55555556)};
   t10.error = t9;
   return t10;
  }
  goto zig_block_1;
 }
 goto zig_block_1;

 zig_block_1:;
 t13.error = UINT16_C(0);
 t14 = &t13.payload;
 t15 = (int32_t *)&t14->handle;
 (*t15) = t6;
 return t13;
}

static uint16_t bincode_serializeUnion__anon_2692__4236(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__1919 const a1) {
 struct shared_Command__struct_1922__1922 t4;
 struct shared_Command__struct_1923__1923 t5;
 uint32_t t1;
 uint16_t t2;
 uint8_t t0;
 bool t3;
 t0 = a1.tag;
 t1 = (uint32_t)t0;
 t2 = bincode_serialize__anon_3123__6009(a0, t1);
 if (t2) {
  return t2;
 }
 t0 = a1.tag;
 t3 = t0 == UINT8_C(0);
 if (t3) {
  t4 = a1.payload.ExecutedBenchmark;
  t2 = bincode_serialize__anon_3125__6010(a0, t4);
  if (t2) {
   return t2;
  }
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(1);
 if (t3) {
  t2 = bincode_serialize__anon_3127__6011(a0);
  if (t2) {
   return t2;
  }
  goto zig_block_1;
 }
 goto zig_block_1;

 zig_block_1:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(2);
 if (t3) {
  t2 = bincode_serialize__anon_3127__6011(a0);
  if (t2) {
   return t2;
  }
  goto zig_block_2;
 }
 goto zig_block_2;

 zig_block_2:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(3);
 if (t3) {
  t2 = bincode_serialize__anon_3127__6011(a0);
  if (t2) {
   return t2;
  }
  goto zig_block_3;
 }
 goto zig_block_3;

 zig_block_3:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(4);
 if (t3) {
  t2 = bincode_serialize__anon_3127__6011(a0);
  if (t2) {
   return t2;
  }
  goto zig_block_4;
 }
 goto zig_block_4;

 zig_block_4:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(5);
 if (t3) {
  t5 = a1.payload.SetIntegration;
  t2 = bincode_serialize__anon_3129__6012(a0, t5);
  if (t2) {
   return t2;
  }
  goto zig_block_5;
 }
 goto zig_block_5;

 zig_block_5:;
 t0 = a1.tag;
 t3 = t0 == UINT8_C(6);
 if (t3) {
  t2 = bincode_serialize__anon_3127__6011(a0);
  if (t2) {
   return t2;
  }
  goto zig_block_6;
 }
 goto zig_block_6;

 zig_block_6:;
 return 0;
}

static nav__3933_39 array_list_ArrayListAligned_28u8_2cnull_29_allocatedSlice__3933(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const a0) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const *t1;
 nav__3933_39 const *t2;
 uint8_t *const *t3;
 uintptr_t t4;
 uint8_t *t5;
 nav__3933_39 t6;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t0;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 const *)&t0;
 t2 = (nav__3933_39 const *)&t1->items;
 t3 = &t2->ptr;
 t4 = a0.capacity;
 t5 = (*t3);
 t5 = (uint8_t *)(((uintptr_t)t5) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t6.ptr = t5;
 t6.len = t4;
 return t6;
}

static nav__1143_38 fs_File_write__1143(struct fs_File__546 const a0, nav__1143_41 const a1) {
 nav__1143_38 t1;
 int32_t t0;
 t0 = a0.handle;
 t1 = posix_write__1363(t0, a1);
 return t1;
}

static nav__1544_39 posix_clock_gettime__1544(uint32_t const a0) {
 struct os_linux_timespec__struct_2728__2728 t3;
 struct os_linux_timespec__struct_2728__2728 t0;
 nav__1544_39 t4;
 int t1;
 uint16_t t2;
 t1 = clock_gettime(a0, &t0);
 t2 = posix_errno__anon_3293__6015(t1);
 switch (t2) {
  case UINT16_C(0): {
   t3 = t0;
   t4.payload = t3;
   t4.error = UINT16_C(0);
   return t4;
  }
  case UINT16_C(14): {
   zig_unreachable();
  }
  case UINT16_C(22): {
   return (nav__1544_39){{-(intptr_t)0x5555555555555556,-(intptr_t)0x5555555555555556},zig_error_UnsupportedClock};
  }
  default: {
   t2 = posix_unexpectedErrno__1627(t2);
   t4.payload = (struct os_linux_timespec__struct_2728__2728){-(intptr_t)0x5555555555555556,-(intptr_t)0x5555555555555556};
   t4.error = t2;
   return t4;
  }
 }
}

static nav__1133_38 fs_File_read__1133(struct fs_File__546 const a0, nav__1133_41 const a1) {
 nav__1133_38 t1;
 int32_t t0;
 t0 = a0.handle;
 t1 = posix_read__1355(t0, a1);
 return t1;
}

static nav__4237_39 mem_Allocator_allocWithSizeAndAlignment__anon_2759__4237(struct mem_Allocator__206 const a0, uintptr_t const a1, uintptr_t const a2) {
 struct mem_Allocator__206 const *t1;
 uintptr_t t2;
 uintptr_t t5;
 nav__4237_49 t3;
 struct mem_Allocator__206 t6;
 struct mem_Allocator__206 t0;
 nav__4237_39 t7;
 bool t4;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t3 = math_mul__anon_3314__6016((uintptr_t)1ul, a1);
 t4 = t3.error == UINT16_C(0);
 if (t4) {
  t5 = t3.payload;
  t2 = t5;
  goto zig_block_0;
 }
 return (nav__4237_39){((uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),zig_error_OutOfMemory};

 zig_block_0:;
 t6 = (*t1);
 t7 = mem_Allocator_allocBytesWithAlignment__anon_3316__6017(t6, t2, a2);
 return t7;
}

static nav__4124_38 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_read__4124(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const a0, nav__4124_42 const a1) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t1;
 nav__4124_42 const *t3;
 uintptr_t t4;
 uintptr_t t7;
 uintptr_t t9;
 nav__4124_42 *t5;
 nav__4124_42 t6;
 nav__4124_42 t13;
 nav__4124_42 t2;
 uintptr_t *t8;
 uint64_t t10;
 uint8_t *t11;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t12;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t0;
 nav__4124_38 t14;
 t0 = a0;
 t1 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t0;
 t2 = a1;
 t3 = (nav__4124_42 const *)&t2;
 t4 = a1.len;
 t5 = (nav__4124_42 *)&a0->buffer;
 t6 = (*t5);
 t7 = t6.len;
 t8 = (uintptr_t *)&a0->pos;
 t9 = (*t8);
 t9 = t7 - t9;
 t9 = (t4 < t9) ? t4 : t9;
 t10 = t9;
 t8 = (uintptr_t *)&a0->pos;
 t9 = (*t8);
 t4 = t10;
 t4 = t9 + t4;
 t6 = (*t3);
 t11 = t6.ptr;
 t11 = (uint8_t *)(((uintptr_t)t11) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t9 = t10;
 t6.ptr = t11;
 t6.len = t9;
 t12 = (*t1);
 t5 = (nav__4124_42 *)&t12->buffer;
 t8 = (uintptr_t *)&a0->pos;
 t9 = (*t8);
 t13 = (*t5);
 t11 = t13.ptr;
 t11 = (uint8_t *)(((uintptr_t)t11) + (t9*sizeof(uint8_t)));
 t9 = t4 - t9;
 t13.ptr = t11;
 t13.len = t9;
 t11 = t13.ptr;
 if (t6.len != 0) memcpy(t6.ptr, t11, t6.len * sizeof(uint8_t));
 t12 = (*t1);
 t8 = (uintptr_t *)&t12->pos;
 (*t8) = t4;
 t4 = t10;
 t14.payload = t4;
 t14.error = UINT16_C(0);
 return t14;
}

static uint16_t io_Reader_readNoEof__4174(struct io_Reader__2282 const a0, nav__4174_40 const a1) {
 struct io_Reader__2282 const *t1;
 struct io_Reader__2282 t2;
 struct io_Reader__2282 t0;
 nav__4174_43 t3;
 uintptr_t t5;
 uintptr_t t6;
 uint64_t t7;
 uint64_t t8;
 uint16_t t4;
 bool t9;
 t0 = a0;
 t1 = (struct io_Reader__2282 const *)&t0;
 t2 = (*t1);
 t3 = io_Reader_readAll__4172(t2, a1);
 if (t3.error) {
  t4 = t3.error;
  return t4;
 }
 t5 = t3.payload;
 t6 = a1.len;
 t7 = t5;
 t8 = t6;
 t9 = t7 < t8;
 if (t9) {
  return zig_error_EndOfStream;
 }
 goto zig_block_0;

 zig_block_0:;
 return 0;
}

static nav__4173_38 io_Reader_readAtLeast__4173(struct io_Reader__2282 const a0, nav__4173_41 const a1, uintptr_t const a2) {
 struct io_Reader__2282 const *t1;
 nav__4173_41 const *t3;
 uintptr_t t4;
 uintptr_t t12;
 uintptr_t t8;
 uint64_t t5;
 uint64_t t6;
 struct io_Reader__2282 t9;
 struct io_Reader__2282 t0;
 nav__4173_41 t10;
 nav__4173_41 t2;
 uint8_t *t11;
 nav__4173_38 t13;
 uint16_t t14;
 bool t7;
 t0 = a0;
 t1 = (struct io_Reader__2282 const *)&t0;
 t2 = a1;
 t3 = (nav__4173_41 const *)&t2;
 t4 = a1.len;
 t5 = a2;
 t6 = t4;
 t7 = t5 <= t6;
 debug_assert__179(t7);
 t8 = (uintptr_t)0ul;
 zig_loop_17:
 t4 = t8;
 t6 = t4;
 t5 = a2;
 t7 = t6 < t5;
 if (t7) {
  t9 = (*t1);
  t4 = t8;
  t10 = (*t3);
  t11 = t10.ptr;
  t11 = (uint8_t *)(((uintptr_t)t11) + (t4*sizeof(uint8_t)));
  t12 = t10.len;
  t4 = t12 - t4;
  t10.ptr = t11;
  t10.len = t4;
  t13 = io_Reader_read__4171(t9, t10);
  if (t13.error) {
   t14 = t13.error;
   t13.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
   t13.error = t14;
   return t13;
  }
  t4 = t13.payload;
  t5 = t4;
  t7 = t5 == UINT64_C(0);
  if (t7) {
   goto zig_block_0;
  }
  goto zig_block_2;

  zig_block_2:;
  t12 = t8;
  t4 = t12 + t4;
  t8 = t4;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_17;

 zig_block_0:;
 t12 = t8;
 t13.payload = t12;
 t13.error = UINT16_C(0);
 return t13;
}

static zig_cold zig_noreturn void debug_no_panic_call__4239(nav__4239_39 const a0, nav__4239_40 const a1) {
 (void)a0;
 (void)a1;
 zig_trap();
}

static nav__4266_39 mem_sliceAsBytes__anon_2876__4266(nav__4266_39 const a0) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t *t4;
 uint8_t *t5;
 uint8_t *const *t6;
 nav__4266_39 t7;
 bool t2;
 bool t3;
 t0 = a0.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  t3 = true;
  goto zig_block_1;
 }
 t3 = false;
 goto zig_block_1;

 zig_block_1:;
 if (t3) {
  return (nav__4266_39){(uint8_t *)((void const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),(uintptr_t)0ul};
 }
 goto zig_block_0;

 zig_block_0:;
 t4 = a0.ptr;
 t5 = t4;
 t6 = (uint8_t *const *)&t5;
 t0 = a0.len;
 t4 = (*t6);
 t4 = (uint8_t *)(((uintptr_t)t4) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t7.ptr = t4;
 t7.len = t0;
 return t7;
}

static nav__5112_38 math_cast__anon_2915__5112(uint64_t const a0) {
 long t1;
 nav__5112_38 t2;
 bool t0;
 t0 = a0 > UINT64_C(9223372036854775807);
 if (t0) {
  return (nav__5112_38){-0x5555555555555556l,true};
 }
 t1 = (long)a0;
 t2.is_null = false;
 t2.payload = t1;
 return t2;
}

static void mem_Allocator_free__anon_2982__5883(struct mem_Allocator__206 const a0, nav__5883_40 const a1) {
 struct mem_Allocator__206 const *t1;
 nav__5883_40 t2;
 uintptr_t t3;
 uint64_t t4;
 uint8_t const *t6;
 uint8_t *t7;
 uint8_t *t8;
 uint8_t *const *t9;
 nav__5883_51 t10;
 struct mem_Allocator__206 t11;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t13;
 struct mem_Allocator_VTable__209 const *const *t14;
 struct mem_Allocator_VTable__209 const *t15;
 void (*const *t16)(void *, nav__5883_51, uint8_t, uintptr_t);
 void (*t17)(void *, nav__5883_51, uint8_t, uintptr_t);
 void *t18;
 bool t5;
 uint8_t t12;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = mem_sliceAsBytes__anon_3347__6018(a1);
 t3 = t2.len;
 t4 = t3;
 t5 = t4 == UINT64_C(0);
 if (t5) {
  return;
 }
 goto zig_block_0;

 zig_block_0:;
 t6 = t2.ptr;
 t7 = (uint8_t *)t6;
 t8 = t7;
 t9 = (uint8_t *const *)&t8;
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t10.ptr = t7;
 t10.len = t3;
 t11 = (*t1);
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t10.ptr = t7;
 t10.len = t3;
 t12 = mem_Alignment_fromByteUnits__628((uintptr_t)1ul);
 t3 = (uintptr_t)zig_return_address();
 t13 = t11;
 t1 = (struct mem_Allocator__206 const *)&t13;
 t14 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t15 = (*t14);
 t16 = (void (*const *)(void *, nav__5883_51, uint8_t, uintptr_t))&t15->free;
 t17 = (*t16);
 t18 = t11.ptr;
 t17(t18, t10, t12, t3);
 return;
}

static bool math_isPowerOfTwo__anon_2984__5884(uintptr_t const a0) {
 uint64_t t0;
 uintptr_t t2;
 bool t1;
 t0 = a0;
 t1 = t0 > UINT64_C(0);
 debug_assert__179(t1);
 t2 = a0 - (uintptr_t)1ul;
 t2 = a0 & t2;
 t0 = t2;
 t1 = t0 == UINT64_C(0);
 return t1;
}

static uint16_t posix_faccessatZ__1519(int32_t const a0, uint8_t const *const a1, uint32_t const a2, uint32_t const a3) {
 unsigned int t0;
 unsigned int t1;
 int t2;
 uint16_t t3;
 t0 = a2;
 t1 = a3;
 t2 = faccessat(a0, a1, t0, t1);
 t3 = posix_errno__anon_3293__6015(t2);
 switch (t3) {
  case UINT16_C(0): {
   return 0;
  }
  case UINT16_C(13): {
   return zig_error_PermissionDenied;
  }
  case UINT16_C(1): {
   return zig_error_PermissionDenied;
  }
  case UINT16_C(30): {
   return zig_error_ReadOnlyFileSystem;
  }
  case UINT16_C(40): {
   return zig_error_SymLinkLoop;
  }
  case UINT16_C(26): {
   return zig_error_FileBusy;
  }
  case UINT16_C(20): {
   return zig_error_FileNotFound;
  }
  case UINT16_C(2): {
   return zig_error_FileNotFound;
  }
  case UINT16_C(36): {
   return zig_error_NameTooLong;
  }
  case UINT16_C(22): {
   zig_unreachable();
  }
  case UINT16_C(14): {
   zig_unreachable();
  }
  case UINT16_C(5): {
   return zig_error_InputOutput;
  }
  case UINT16_C(12): {
   return zig_error_SystemResources;
  }
  case UINT16_C(84): {
   t3 = posix_unexpectedErrno__1627(t3);
   return t3;
  }
  default: {
   t3 = posix_unexpectedErrno__1627(t3);
   return t3;
  }
 }
}

static nav__1375_38 posix_openatZ__1375(int32_t const a0, uint8_t const *const a1, uint32_t const a2, uintptr_t const a3) {
 unsigned long t1;
 int t0;
 int32_t t3;
 nav__1375_38 t4;
 uint16_t t2;
 zig_loop_11:
 t0 = a0;
 t1 = a3;
 t0 = openat64(t0, a1, a2, t1);
 t2 = posix_errno__anon_3293__6015(t0);
 switch (t2) {
  case UINT16_C(0): {
   t3 = t0;
   t4.payload = t3;
   t4.error = UINT16_C(0);
   return t4;
  }
  case UINT16_C(4): {
   goto zig_block_0;
  }
  case UINT16_C(14): {
   zig_unreachable();
  }
  case UINT16_C(22): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_BadPathName};
  }
  case UINT16_C(9): {
   zig_unreachable();
  }
  case UINT16_C(13): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_AccessDenied};
  }
  case UINT16_C(27): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_FileTooBig};
  }
  case UINT16_C(75): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_FileTooBig};
  }
  case UINT16_C(21): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_IsDir};
  }
  case UINT16_C(40): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_SymLinkLoop};
  }
  case UINT16_C(24): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_ProcessFdQuotaExceeded};
  }
  case UINT16_C(36): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_NameTooLong};
  }
  case UINT16_C(23): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_SystemFdQuotaExceeded};
  }
  case UINT16_C(19): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_NoDevice};
  }
  case UINT16_C(2): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_FileNotFound};
  }
  case UINT16_C(12): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_SystemResources};
  }
  case UINT16_C(28): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_NoSpaceLeft};
  }
  case UINT16_C(20): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_NotDir};
  }
  case UINT16_C(1): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_AccessDenied};
  }
  case UINT16_C(17): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_PathAlreadyExists};
  }
  case UINT16_C(16): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_DeviceBusy};
  }
  case UINT16_C(95): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_FileLocksNotSupported};
  }
  case UINT16_C(11): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_WouldBlock};
  }
  case UINT16_C(26): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_FileBusy};
  }
  case UINT16_C(6): {
   return (nav__1375_38){-INT32_C(0x55555556),zig_error_NoDevice};
  }
  case UINT16_C(84): {
   t2 = posix_unexpectedErrno__1627(t2);
   t4.payload = -INT32_C(0x55555556);
   t4.error = t2;
   return t4;
  }
  default: {
   t2 = posix_unexpectedErrno__1627(t2);
   t4.payload = -INT32_C(0x55555556);
   t4.error = t2;
   return t4;
  }
 }

 zig_block_0:;
 goto zig_loop_11;
}

static uint16_t posix_flock__1536(int32_t const a0, int32_t const a1) {
 int t0;
 uint16_t t1;
 zig_loop_3:
 t0 = a1;
 t0 = flock(a0, t0);
 t1 = posix_errno__anon_3293__6015(t0);
 switch (t1) {
  case UINT16_C(0): {
   return 0;
  }
  case UINT16_C(9): {
   zig_unreachable();
  }
  case UINT16_C(4): {
   goto zig_block_0;
  }
  case UINT16_C(22): {
   zig_unreachable();
  }
  case UINT16_C(37): {
   return zig_error_SystemResources;
  }
  case UINT16_C(11): {
   return zig_error_WouldBlock;
  }
  case UINT16_C(95): {
   return zig_error_FileLocksNotSupported;
  }
  default: {
   t1 = posix_unexpectedErrno__1627(t1);
   return t1;
  }
 }

 zig_block_0:;
 goto zig_loop_3;
}

static void posix_close__1333(int32_t const a0) {
 int t0;
 uint16_t t1;
 t0 = close(a0);
 t1 = posix_errno__anon_3293__6015(t0);
 switch (t1) {
  case UINT16_C(9): {
   zig_unreachable();
  }
  case UINT16_C(4): {
   return;
  }
  default: {
   return;
  }
 }
}

static uint16_t bincode_serialize__anon_3123__6009(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, uint32_t const a1) {
 uint16_t t0;
 t0 = bincode_serializeInt__anon_3509__6022(a0, a1);
 if (t0) {
  return t0;
 }
 return 0;
}

static uint16_t bincode_serialize__anon_3125__6010(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__struct_1922__1922 const a1) {
 uint16_t t0;
 t0 = bincode_serializeStruct__anon_3514__6023(a0, a1);
 if (t0) {
  return t0;
 }
 return 0;
}

static uint16_t bincode_serialize__anon_3127__6011(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0) {
 (void)a0;
 return 0;
}

static uint16_t bincode_serialize__anon_3129__6012(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__struct_1923__1923 const a1) {
 uint16_t t0;
 t0 = bincode_serializeStruct__anon_3515__6024(a0, a1);
 if (t0) {
  return t0;
 }
 return 0;
}

static nav__1363_38 posix_write__1363(int32_t const a0, nav__1363_40 const a1) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t const *t3;
 intptr_t t5;
 nav__1363_38 t7;
 uint32_t t4;
 uint16_t t6;
 bool t2;
 t0 = a1.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  return (nav__1363_38){(uintptr_t)0ul,0};
 }
 goto zig_block_0;

 zig_block_0:;
 zig_loop_16:
 t3 = a1.ptr;
 t0 = a1.len;
 t0 = ((uintptr_t)2147479552ul < t0) ? (uintptr_t)2147479552ul : t0;
 t4 = (uint32_t)t0;
 t0 = (uintptr_t)t4;
 t5 = write(a0, t3, t0);
 t6 = posix_errno__anon_3526__6026(t5);
 switch (t6) {
  case UINT16_C(0): {
   t0 = (uintptr_t)t5;
   t7.payload = t0;
   t7.error = UINT16_C(0);
   return t7;
  }
  case UINT16_C(4): {
   goto zig_block_1;
  }
  case UINT16_C(22): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_InvalidArgument};
  }
  case UINT16_C(14): {
   zig_unreachable();
  }
  case UINT16_C(2): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_ProcessNotFound};
  }
  case UINT16_C(11): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_WouldBlock};
  }
  case UINT16_C(9): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NotOpenForWriting};
  }
  case UINT16_C(89): {
   zig_unreachable();
  }
  case UINT16_C(122): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_DiskQuota};
  }
  case UINT16_C(27): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_FileTooBig};
  }
  case UINT16_C(5): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_InputOutput};
  }
  case UINT16_C(28): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NoSpaceLeft};
  }
  case UINT16_C(13): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_AccessDenied};
  }
  case UINT16_C(1): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_AccessDenied};
  }
  case UINT16_C(32): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_BrokenPipe};
  }
  case UINT16_C(104): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_ConnectionResetByPeer};
  }
  case UINT16_C(16): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_DeviceBusy};
  }
  case UINT16_C(6): {
   return (nav__1363_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NoDevice};
  }
  default: {
   t6 = posix_unexpectedErrno__1627(t6);
   t7.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
   t7.error = t6;
   return t7;
  }
 }

 zig_block_1:;
 goto zig_loop_16;
}

static uint16_t posix_errno__anon_3293__6015(int const a0) {
 int *t3;
 int32_t t1;
 int t4;
 uint16_t t0;
 uint16_t t5;
 bool t2;
 t1 = a0;
 t2 = t1 == -INT32_C(1);
 if (t2) {
  t3 = zig_e___errno_location();
  t4 = (*t3);
  t5 = (uint16_t)t4;
  t0 = t5;
  goto zig_block_0;
 }
 t0 = UINT16_C(0);
 goto zig_block_0;

 zig_block_0:;
 return t0;
}

static uint16_t posix_unexpectedErrno__1627(uint16_t const a0) {
 (void)a0;
 return zig_error_Unexpected;
}

static nav__1355_38 posix_read__1355(int32_t const a0, nav__1355_40 const a1) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t *t3;
 intptr_t t5;
 nav__1355_38 t7;
 uint32_t t4;
 uint16_t t6;
 bool t2;
 t0 = a1.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  return (nav__1355_38){(uintptr_t)0ul,0};
 }
 goto zig_block_0;

 zig_block_0:;
 zig_loop_16:
 t3 = a1.ptr;
 t0 = a1.len;
 t0 = ((uintptr_t)2147479552ul < t0) ? (uintptr_t)2147479552ul : t0;
 t4 = (uint32_t)t0;
 t0 = (uintptr_t)t4;
 t5 = read(a0, t3, t0);
 t6 = posix_errno__anon_3526__6026(t5);
 switch (t6) {
  case UINT16_C(0): {
   t0 = (uintptr_t)t5;
   t7.payload = t0;
   t7.error = UINT16_C(0);
   return t7;
  }
  case UINT16_C(4): {
   goto zig_block_1;
  }
  case UINT16_C(22): {
   zig_unreachable();
  }
  case UINT16_C(14): {
   zig_unreachable();
  }
  case UINT16_C(2): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_ProcessNotFound};
  }
  case UINT16_C(11): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_WouldBlock};
  }
  case UINT16_C(125): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_Canceled};
  }
  case UINT16_C(9): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NotOpenForReading};
  }
  case UINT16_C(5): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_InputOutput};
  }
  case UINT16_C(21): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_IsDir};
  }
  case UINT16_C(105): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_SystemResources};
  }
  case UINT16_C(12): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_SystemResources};
  }
  case UINT16_C(107): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_SocketNotConnected};
  }
  case UINT16_C(104): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_ConnectionResetByPeer};
  }
  case UINT16_C(110): {
   return (nav__1355_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_ConnectionTimedOut};
  }
  default: {
   t6 = posix_unexpectedErrno__1627(t6);
   t7.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
   t7.error = t6;
   return t7;
  }
 }

 zig_block_1:;
 goto zig_loop_16;
}

static nav__6016_38 math_mul__anon_3314__6016(uintptr_t const a0, uintptr_t const a1) {
 nav__6016_42 t0;
 uintptr_t t3;
 nav__6016_38 t4;
 uint8_t t1;
 bool t2;
 t0.f1 = zig_mulo_u64(&t0.f0, a0, a1, UINT8_C(64));
 t1 = t0.f1;
 t2 = t1 != UINT8_C(0);
 if (t2) {
  return (nav__6016_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_Overflow};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = t0.f0;
 t4.payload = t3;
 t4.error = UINT16_C(0);
 return t4;
}

static nav__6017_39 mem_Allocator_allocBytesWithAlignment__anon_3316__6017(struct mem_Allocator__206 const a0, uintptr_t const a1, uintptr_t const a2) {
 struct mem_Allocator__206 const *t1;
 uint64_t t2;
 uint8_t *t4;
 uint8_t *t13;
 uint8_t *t14;
 uint8_t *t15;
 struct mem_Allocator__206 t5;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t7;
 struct mem_Allocator_VTable__209 const *const *t8;
 struct mem_Allocator_VTable__209 const *t9;
 uint8_t *(*const *t10)(void *, uintptr_t, uint8_t, uintptr_t);
 uint8_t *(*t11)(void *, uintptr_t, uint8_t, uintptr_t);
 void *t12;
 uint8_t *const *t16;
 nav__6017_52 t17;
 nav__6017_39 t18;
 bool t3;
 uint8_t t6;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = a1;
 t3 = t2 == UINT64_C(0);
 if (t3) {
  return (nav__6017_39){(uint8_t *)UINTPTR_MAX,0};
 }
 goto zig_block_0;

 zig_block_0:;
 t5 = (*t1);
 t6 = mem_Alignment_fromByteUnits__628((uintptr_t)1ul);
 t7 = t5;
 t1 = (struct mem_Allocator__206 const *)&t7;
 t8 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t9 = (*t8);
 t10 = (uint8_t *(*const *)(void *, uintptr_t, uint8_t, uintptr_t))&t9->alloc;
 t11 = (*t10);
 t12 = t5.ptr;
 t13 = t11(t12, a1, t6, a2);
 t3 = t13 != NULL;
 if (t3) {
  t14 = t13;
  t4 = t14;
  goto zig_block_1;
 }
 return (nav__6017_39){((uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),zig_error_OutOfMemory};

 zig_block_1:;
 t15 = t4;
 t16 = (uint8_t *const *)&t15;
 t14 = (*t16);
 t14 = (uint8_t *)(((uintptr_t)t14) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t17.ptr = t14;
 t17.len = a1;
 t18.payload = t4;
 t18.error = UINT16_C(0);
 return t18;
}

static nav__4171_38 io_Reader_read__4171(struct io_Reader__2282 const a0, nav__4171_41 const a1) {
 struct io_Reader__2282 const *t1;
 nav__4171_38 (*const *t2)(void const *, nav__4171_41);
 nav__4171_38 (*t3)(void const *, nav__4171_41);
 void const *t4;
 nav__4171_38 t5;
 struct io_Reader__2282 t0;
 t0 = a0;
 t1 = (struct io_Reader__2282 const *)&t0;
 t2 = (nav__4171_38 (*const *)(void const *, nav__4171_41))&t1->readFn;
 t3 = (*t2);
 t4 = a0.context;
 t5 = t3(t4, a1);
 return t5;
}

static nav__6018_39 mem_sliceAsBytes__anon_3347__6018(nav__6018_39 const a0) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t const *t4;
 uint8_t const *t5;
 uint8_t const *const *t6;
 nav__6018_39 t7;
 bool t2;
 bool t3;
 t0 = a0.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  t3 = true;
  goto zig_block_1;
 }
 t3 = false;
 goto zig_block_1;

 zig_block_1:;
 if (t3) {
  return (nav__6018_39){(uint8_t const *)((void const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),(uintptr_t)0ul};
 }
 goto zig_block_0;

 zig_block_0:;
 t4 = a0.ptr;
 t5 = t4;
 t6 = (uint8_t const *const *)&t5;
 t0 = a0.len;
 t4 = (*t6);
 t4 = (uint8_t const *)(((uintptr_t)t4) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t7.ptr = t4;
 t7.len = t0;
 return t7;
}

static uint16_t bincode_serializeInt__anon_3509__6022(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, uint32_t const a1) {
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t1;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t4;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t2;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t0;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t3;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *const *t5;
 void const **t7;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t8;
 void const *t9;
 nav__6022_48 (**t10)(void const *, nav__6022_50);
 struct io_Writer__3628 t11;
 struct io_Writer__3628 t6;
 struct io_Writer__3628 t12;
 struct io_Writer__3628 t14;
 struct io_Writer__3628 const *t13;
 nav__6022_50 t17;
 uint16_t t18;
 uint16_t t19;
 uint8_t t16[4];
 uint8_t t15[4];
 t0 = a0;
 t1 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *)&t0;
 t2 = (*t1);
 t3 = t2;
 t1 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *)&t3;
 t4 = t1;
 t5 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *const *)&t4;
 t7 = (void const **)&t6.context;
 t1 = (*t5);
 t8 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t1->context;
 t9 = (void const *)t8;
 (*t7) = t9;
 t10 = (nav__6022_48 (**)(void const *, nav__6022_50))&t6.writeFn;
 (*t10) = &io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29___4057;
 t11 = t6;
 t12 = t11;
 t13 = (struct io_Writer__3628 const *)&t12;
 t11 = (*t13);
 t14 = t11;
 t13 = (struct io_Writer__3628 const *)&t14;
 memcpy(&t16, &a1, sizeof(uint8_t[4]));
 memcpy((char *)&t15, t16, sizeof(uint8_t[4]));
 t11 = (*t13);
 t17.ptr = &t15[(uintptr_t)0ul];
 t17.len = (uintptr_t)4ul;
 t18 = io_Writer_writeAll__6036(t11, t17);
 memcpy(&t19, &t18, sizeof(uint16_t));
 if (t19) {
  return t19;
 }
 return 0;
}

static uint16_t bincode_serializeStruct__anon_3514__6023(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__struct_1922__1922 const a1) {
 nav__6023_45 t2;
 uint32_t t0;
 uint16_t t1;
 t0 = a1.pid;
 t1 = bincode_serialize__anon_3123__6009(a0, t0);
 if (t1) {
  return t1;
 }
 t2 = a1.uri;
 t1 = bincode_serialize__anon_3664__6045(a0, t2);
 if (t1) {
  return t1;
 }
 return 0;
}

static uint16_t bincode_serializeStruct__anon_3515__6024(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, struct shared_Command__struct_1923__1923 const a1) {
 nav__6024_45 t0;
 uint16_t t1;
 t0 = a1.name;
 t1 = bincode_serialize__anon_3664__6045(a0, t0);
 if (t1) {
  return t1;
 }
 t0 = a1.version;
 t1 = bincode_serialize__anon_3664__6045(a0, t0);
 if (t1) {
  return t1;
 }
 return 0;
}

static uint16_t posix_errno__anon_3526__6026(intptr_t const a0) {
 int64_t t1;
 int *t3;
 int t4;
 uint16_t t0;
 uint16_t t5;
 bool t2;
 t1 = a0;
 t2 = t1 == -INT64_C(1);
 if (t2) {
  t3 = zig_e___errno_location();
  t4 = (*t3);
  t5 = (uint16_t)t4;
  t0 = t5;
  goto zig_block_0;
 }
 t0 = UINT16_C(0);
 goto zig_block_0;

 zig_block_0:;
 return t0;
}

static nav__4057_38 io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29___4057(void const *const a0, nav__4057_41 const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t0;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t1;
 nav__4057_38 t2;
 nav__4057_38 t3;
 t0 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)a0;
 t1 = (*t0);
 t2 = array_list_ArrayListAligned_28u8_2cnull_29_appendWrite__3911(t1, a1);
 memcpy(&t3, &t2, sizeof(nav__4057_38));
 return t3;
}

static uint16_t io_Writer_writeAll__6036(struct io_Writer__3628 const a0, nav__6036_40 const a1) {
 struct io_Writer__3628 const *t1;
 nav__6036_40 const *t3;
 uintptr_t t5;
 uintptr_t t6;
 uintptr_t t13;
 uintptr_t t4;
 uint64_t t7;
 uint64_t t8;
 struct io_Writer__3628 t10;
 struct io_Writer__3628 t0;
 nav__6036_40 t11;
 nav__6036_40 t2;
 uint8_t const *t12;
 nav__6036_43 t14;
 uint16_t t15;
 bool t9;
 t0 = a0;
 t1 = (struct io_Writer__3628 const *)&t0;
 t2 = a1;
 t3 = (nav__6036_40 const *)&t2;
 t4 = (uintptr_t)0ul;
 zig_loop_11:
 t5 = t4;
 t6 = a1.len;
 t7 = t5;
 t8 = t6;
 t9 = t7 != t8;
 if (t9) {
  t6 = t4;
  t10 = (*t1);
  t5 = t4;
  t11 = (*t3);
  t12 = t11.ptr;
  t12 = (uint8_t const *)(((uintptr_t)t12) + (t5*sizeof(uint8_t)));
  t13 = t11.len;
  t5 = t13 - t5;
  t11.ptr = t12;
  t11.len = t5;
  t14 = io_Writer_write__6035(t10, t11);
  if (t14.error) {
   t15 = t14.error;
   return t15;
  }
  t5 = t14.payload;
  t5 = t6 + t5;
  t4 = t5;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_11;

 zig_block_0:;
 return 0;
}

static uint16_t bincode_serialize__anon_3664__6045(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, nav__6045_40 const a1) {
 uint16_t t0;
 t0 = bincode_serializePointer__anon_3676__6046(a0, a1);
 if (t0) {
  return t0;
 }
 return 0;
}

static nav__3911_38 array_list_ArrayListAligned_28u8_2cnull_29_appendWrite__3911(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, nav__3911_42 const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t2;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 nav__3911_38 t4;
 uintptr_t t5;
 uint16_t t3;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (*t1);
 t3 = array_list_ArrayListAligned_28u8_2cnull_29_appendSlice__3905(t2, a1);
 if (t3) {
  t4.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
  t4.error = t3;
  return t4;
 }
 t5 = a1.len;
 t4.payload = t5;
 t4.error = UINT16_C(0);
 return t4;
}

static nav__6035_38 io_Writer_write__6035(struct io_Writer__3628 const a0, nav__6035_41 const a1) {
 struct io_Writer__3628 const *t1;
 nav__6035_38 (*const *t2)(void const *, nav__6035_41);
 nav__6035_38 (*t3)(void const *, nav__6035_41);
 void const *t4;
 nav__6035_38 t5;
 struct io_Writer__3628 t0;
 t0 = a0;
 t1 = (struct io_Writer__3628 const *)&t0;
 t2 = (nav__6035_38 (*const *)(void const *, nav__6035_41))&t1->writeFn;
 t3 = (*t2);
 t4 = a0.context;
 t5 = t3(t4, a1);
 return t5;
}

static uint16_t bincode_serializePointer__anon_3676__6046(struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const a0, nav__6046_40 const a1) {
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t1;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t6;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t7;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *t24;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t2;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t0;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t5;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 t23;
 uintptr_t t3;
 uint64_t t4;
 struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *const *t8;
 void const **t10;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t11;
 void const *t12;
 nav__6046_51 (**t13)(void const *, nav__6046_40);
 struct io_Writer__3628 t14;
 struct io_Writer__3628 t9;
 struct io_Writer__3628 t15;
 struct io_Writer__3628 t17;
 struct io_Writer__3628 t25;
 struct io_Writer__3628 t26;
 struct io_Writer__3628 const *t16;
 nav__6046_40 t20;
 uint16_t t21;
 uint16_t t22;
 uint8_t t19[8];
 uint8_t t18[8];
 t0 = a0;
 t1 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *)&t0;
 t2 = (*t1);
 t3 = a1.len;
 t4 = t3;
 t5 = t2;
 t6 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *)&t5;
 t7 = t6;
 t8 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *const *)&t7;
 t10 = (void const **)&t9.context;
 t6 = (*t8);
 t11 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t6->context;
 t12 = (void const *)t11;
 (*t10) = t12;
 t13 = (nav__6046_51 (**)(void const *, nav__6046_40))&t9.writeFn;
 (*t13) = &io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29___4057;
 t14 = t9;
 t15 = t14;
 t16 = (struct io_Writer__3628 const *)&t15;
 t14 = (*t16);
 t17 = t14;
 t16 = (struct io_Writer__3628 const *)&t17;
 memcpy(&t19, &t4, sizeof(uint8_t[8]));
 memcpy((char *)&t18, t19, sizeof(uint8_t[8]));
 t14 = (*t16);
 t20.ptr = &t18[(uintptr_t)0ul];
 t20.len = (uintptr_t)8ul;
 t21 = io_Writer_writeAll__6036(t14, t20);
 memcpy(&t22, &t21, sizeof(uint16_t));
 if (t22) {
  return t22;
 }
 t2 = (*t1);
 t23 = t2;
 t1 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *)&t23;
 t24 = t1;
 t8 = (struct io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29__1981 const *const *)&t24;
 t10 = (void const **)&t25.context;
 t1 = (*t8);
 t11 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t1->context;
 t12 = (void const *)t11;
 (*t10) = t12;
 t13 = (nav__6046_51 (**)(void const *, nav__6046_40))&t25.writeFn;
 (*t13) = &io_GenericWriter_28_2aarray_list_ArrayListAligned_28u8_2cnull_29_2cerror_7bOutOfMemory_7d_2c_28function_20_27appendWrite_27_29_29___4057;
 t14 = t25;
 t26 = t14;
 t16 = (struct io_Writer__3628 const *)&t26;
 t14 = (*t16);
 t22 = io_Writer_writeAll__6036(t14, a1);
 memcpy(&t21, &t22, sizeof(uint16_t));
 if (t21) {
  return t21;
 }
 return 0;
}

static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_appendSlice__3905(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, nav__3905_41 const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t2;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 uintptr_t t3;
 uint16_t t4;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (*t1);
 t3 = a1.len;
 t4 = array_list_ArrayListAligned_28u8_2cnull_29_ensureUnusedCapacity__3924(t2, t3);
 if (t4) {
  return t4;
 }
 t2 = (*t1);
 array_list_ArrayListAligned_28u8_2cnull_29_appendSliceAssumeCapacity__3906(t2, a1);
 return 0;
}

static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureUnusedCapacity__3924(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, uintptr_t const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t2;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 nav__3924_43 *t3;
 nav__3924_43 t4;
 uintptr_t t5;
 nav__3924_52 t6;
 uint16_t t7;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (*t1);
 t3 = (nav__3924_43 *)&a0->items;
 t4 = (*t3);
 t5 = t4.len;
 t6 = array_list_addOrOom__3849(t5, a1);
 if (t6.error) {
  t7 = t6.error;
  return t7;
 }
 t5 = t6.payload;
 t7 = array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacity__3922(t2, t5);
 return t7;
}

static void array_list_ArrayListAligned_28u8_2cnull_29_appendSliceAssumeCapacity__3906(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, nav__3906_41 const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 nav__3906_47 *t2;
 nav__3906_47 t3;
 uintptr_t t4;
 uintptr_t t5;
 uintptr_t t7;
 uintptr_t *t6;
 uint64_t t8;
 uint64_t t9;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t11;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 uint8_t *t12;
 uint8_t const *t13;
 bool t10;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (nav__3906_47 *)&a0->items;
 t3 = (*t2);
 t4 = t3.len;
 t5 = a1.len;
 t5 = t4 + t5;
 t6 = (uintptr_t *)&a0->capacity;
 t7 = (*t6);
 t8 = t5;
 t9 = t7;
 t10 = t8 <= t9;
 debug_assert__179(t10);
 t11 = (*t1);
 t2 = (nav__3906_47 *)&t11->items;
 t6 = &t2->len;
 (*t6) = t5;
 t11 = (*t1);
 t2 = (nav__3906_47 *)&t11->items;
 t5 = a1.len;
 t3 = (*t2);
 t12 = t3.ptr;
 t12 = (uint8_t *)(((uintptr_t)t12) + (t4*sizeof(uint8_t)));
 t3.ptr = t12;
 t3.len = t5;
 t13 = a1.ptr;
 if (t3.len != 0) memcpy(t3.ptr, t13, t3.len * sizeof(uint8_t));
 return;
}

static nav__3849_38 array_list_addOrOom__3849(uintptr_t const a0, uintptr_t const a1) {
 nav__3849_42 t2;
 uintptr_t t3;
 uintptr_t t0;
 uintptr_t const *t5;
 uint8_t const *t6;
 nav__3849_38 t8;
 uint8_t t4;
 uint8_t t1;
 bool t7;
 t2.f1 = zig_addo_u64(&t2.f0, a0, a1, UINT8_C(64));
 t3 = t2.f0;
 t0 = t3;
 t4 = t2.f1;
 t1 = t4;
 t5 = (uintptr_t const *)&t0;
 t6 = (uint8_t const *)&t1;
 t4 = (*t6);
 t7 = t4 != UINT8_C(0);
 if (t7) {
  return (nav__3849_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_OutOfMemory};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = (*t5);
 t8.payload = t3;
 t8.error = UINT16_C(0);
 return t8;
}

static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacity__3922(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, uintptr_t const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 uintptr_t *t2;
 uintptr_t t3;
 uint64_t t4;
 uint64_t t5;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t7;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 uint16_t t8;
 bool t6;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (uintptr_t *)&a0->capacity;
 t3 = (*t2);
 t4 = t3;
 t5 = a1;
 t6 = t4 >= t5;
 if (t6) {
  return 0;
 }
 goto zig_block_0;

 zig_block_0:;
 t2 = (uintptr_t *)&a0->capacity;
 t3 = (*t2);
 t3 = array_list_ArrayListAlignedUnmanaged_28u8_2cnull_29_growCapacity__6105(t3, a1);
 t7 = (*t1);
 t8 = array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacityPrecise__3923(t7, t3);
 return t8;
}

static uintptr_t array_list_ArrayListAlignedUnmanaged_28u8_2cnull_29_growCapacity__6105(uintptr_t const a0, uintptr_t const a1) {
 uintptr_t t1;
 uintptr_t t2;
 uintptr_t t0;
 uint64_t t3;
 uint64_t t4;
 bool t5;
 t0 = a0;
 zig_loop_6:
 t1 = t0;
 t2 = t0;
 t2 = t2 / (uintptr_t)2ul;
 t2 = t2 + (uintptr_t)128ul;
 t2 = zig_adds_u64(t1, t2, UINT8_C(64));
 t0 = t2;
 t2 = t0;
 t3 = t2;
 t4 = a1;
 t5 = t3 >= t4;
 if (t5) {
  t2 = t0;
  return t2;
 }
 goto zig_block_0;

 zig_block_0:;
 goto zig_loop_6;
}

static uint16_t array_list_ArrayListAligned_28u8_2cnull_29_ensureTotalCapacityPrecise__3923(struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const a0, uintptr_t const a1) {
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *t1;
 uintptr_t *t2;
 uintptr_t t3;
 uint64_t t4;
 uint64_t t5;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t7;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *t0;
 struct array_list_ArrayListAligned_28u8_2cnull_29__1950 t8;
 nav__3923_43 t9;
 nav__3923_43 t12;
 nav__3923_43 t20;
 nav__3923_43 t21;
 nav__3923_43 t18;
 struct mem_Allocator__206 *t10;
 struct mem_Allocator__206 t11;
 nav__3923_43 *t13;
 uint8_t **t14;
 uint8_t *t15;
 nav__3923_55 t16;
 nav__3923_43 const *t19;
 uint16_t t17;
 bool t6;
 t0 = a0;
 t1 = (struct array_list_ArrayListAligned_28u8_2cnull_29__1950 *const *)&t0;
 t2 = (uintptr_t *)&a0->capacity;
 t3 = (*t2);
 t4 = t3;
 t5 = a1;
 t6 = t4 >= t5;
 if (t6) {
  return 0;
 }
 goto zig_block_0;

 zig_block_0:;
 t7 = (*t1);
 t8 = (*t7);
 t9 = array_list_ArrayListAligned_28u8_2cnull_29_allocatedSlice__3933(t8);
 t7 = (*t1);
 t10 = (struct mem_Allocator__206 *)&t7->allocator;
 t11 = (*t10);
 t12 = mem_Allocator_remap__anon_3736__6132(t11, t9, a1);
 t6 = t12.ptr != NULL;
 if (t6) {
  t9 = t12;
  t7 = (*t1);
  t13 = (nav__3923_43 *)&t7->items;
  t14 = &t13->ptr;
  t15 = t9.ptr;
  (*t14) = t15;
  t7 = (*t1);
  t2 = (uintptr_t *)&t7->capacity;
  t3 = t9.len;
  (*t2) = t3;
  goto zig_block_1;
 }
 t7 = (*t1);
 t10 = (struct mem_Allocator__206 *)&t7->allocator;
 t11 = (*t10);
 t16 = mem_Allocator_alignedAlloc__anon_3741__6133(t11, a1);
 if (t16.error) {
  t17 = t16.error;
  return t17;
 }
 t12 = t16.payload;
 t18 = t12;
 t19 = (nav__3923_43 const *)&t18;
 t13 = (nav__3923_43 *)&a0->items;
 t20 = (*t13);
 t3 = t20.len;
 t20 = (*t19);
 t15 = t20.ptr;
 t15 = (uint8_t *)(((uintptr_t)t15) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t20.ptr = t15;
 t20.len = t3;
 t13 = (nav__3923_43 *)&a0->items;
 t21 = (*t13);
 t15 = t21.ptr;
 if (t20.len != 0) memcpy(t20.ptr, t15, t20.len * sizeof(uint8_t));
 t7 = (*t1);
 t10 = (struct mem_Allocator__206 *)&t7->allocator;
 t11 = (*t10);
 mem_Allocator_free__anon_2541__4217(t11, t9);
 t7 = (*t1);
 t13 = (nav__3923_43 *)&t7->items;
 t14 = &t13->ptr;
 t15 = t12.ptr;
 (*t14) = t15;
 t7 = (*t1);
 t2 = (uintptr_t *)&t7->capacity;
 t3 = t12.len;
 (*t2) = t3;
 goto zig_block_1;

 zig_block_1:;
 return 0;
}

static nav__6132_39 mem_Allocator_remap__anon_3736__6132(struct mem_Allocator__206 const a0, nav__6132_39 const a1, uintptr_t const a2) {
 struct mem_Allocator__206 const *t1;
 nav__6132_39 const *t3;
 uint64_t t4;
 struct mem_Allocator__206 t6;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t15;
 nav__6132_39 t7;
 nav__6132_39 t10;
 nav__6132_39 t2;
 uint8_t *t8;
 uint8_t *t20;
 uint8_t *t21;
 uint8_t *t22;
 void *t9;
 uintptr_t t11;
 uintptr_t t13;
 nav__6132_50 t12;
 struct mem_Allocator_VTable__209 const *const *t16;
 struct mem_Allocator_VTable__209 const *t17;
 uint8_t *(*const *t18)(void *, nav__6132_39, uint8_t, uintptr_t, uintptr_t);
 uint8_t *(*t19)(void *, nav__6132_39, uint8_t, uintptr_t, uintptr_t);
 uint8_t *const *t23;
 bool t5;
 uint8_t t14;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = a1;
 t3 = (nav__6132_39 const *)&t2;
 t4 = a2;
 t5 = t4 == UINT64_C(0);
 if (t5) {
  t6 = (*t1);
  mem_Allocator_free__anon_2541__4217(t6, a1);
  t7 = (*t3);
  t8 = t7.ptr;
  t8 = (uint8_t *)(((uintptr_t)t8) + ((uintptr_t)0ul*sizeof(uint8_t)));
  t9 = (void *)t8;
  t7.ptr = t9;
  t7.len = (uintptr_t)0ul;
  t10 = t7;
  return t10;
 }
 goto zig_block_0;

 zig_block_0:;
 t11 = a1.len;
 t4 = t11;
 t5 = t4 == UINT64_C(0);
 if (t5) {
  return (nav__6132_39){NULL,(uintptr_t)0xaaaaaaaaaaaaaaaaul};
 }
 goto zig_block_1;

 zig_block_1:;
 t10 = mem_sliceAsBytes__anon_2876__4266(a1);
 t12 = math_mul__anon_3314__6016((uintptr_t)1ul, a2);
 t5 = t12.error == UINT16_C(0);
 if (t5) {
  t13 = t12.payload;
  t11 = t13;
  goto zig_block_2;
 }
 return (nav__6132_39){NULL,(uintptr_t)0xaaaaaaaaaaaaaaaaul};

 zig_block_2:;
 t6 = (*t1);
 t14 = mem_Alignment_fromByteUnits__628((uintptr_t)1ul);
 t13 = (uintptr_t)zig_return_address();
 t15 = t6;
 t1 = (struct mem_Allocator__206 const *)&t15;
 t16 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t17 = (*t16);
 t18 = (uint8_t *(*const *)(void *, nav__6132_39, uint8_t, uintptr_t, uintptr_t))&t17->remap;
 t19 = (*t18);
 t9 = t6.ptr;
 t20 = t19(t9, t10, t14, t11, t13);
 t5 = t20 != NULL;
 if (t5) {
  t21 = t20;
  t8 = t21;
  goto zig_block_3;
 }
 return (nav__6132_39){NULL,(uintptr_t)0xaaaaaaaaaaaaaaaaul};

 zig_block_3:;
 t22 = t8;
 t23 = (uint8_t *const *)&t22;
 t8 = (*t23);
 t8 = (uint8_t *)(((uintptr_t)t8) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t10.ptr = t8;
 t10.len = t11;
 memcpy(&t7, &t10, sizeof(nav__6132_39));
 t7 = mem_bytesAsSlice__anon_3756__6134(t7);
 t10 = t7;
 return t10;
}

static nav__6133_40 mem_Allocator_alignedAlloc__anon_3741__6133(struct mem_Allocator__206 const a0, uintptr_t const a1) {
 struct mem_Allocator__206 const *t1;
 struct mem_Allocator__206 t2;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t5;
 uintptr_t t3;
 nav__6133_40 t4;
 nav__6133_40 t8;
 nav__6133_51 t6;
 uint8_t *t9;
 uint8_t *t10;
 uint8_t *const *t11;
 nav__6133_39 t12;
 uint16_t t7;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = (*t1);
 t3 = (uintptr_t)zig_return_address();
 t5 = t2;
 t1 = (struct mem_Allocator__206 const *)&t5;
 t2 = (*t1);
 t6 = mem_Allocator_allocWithSizeAndAlignment__anon_2759__4237(t2, a1, t3);
 if (t6.error) {
  t7 = t6.error;
  t8.payload = (nav__6133_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t8.error = t7;
  t4 = t8;
  goto zig_block_0;
 }
 t9 = t6.payload;
 t10 = t9;
 t11 = (uint8_t *const *)&t10;
 t9 = (*t11);
 t9 = (uint8_t *)(((uintptr_t)t9) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t12.ptr = t9;
 t12.len = a1;
 t8.payload = t12;
 t8.error = UINT16_C(0);
 t4 = t8;
 goto zig_block_0;

 zig_block_0:;
 return t4;
}

static nav__6134_39 mem_bytesAsSlice__anon_3756__6134(nav__6134_39 const a0) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t *t3;
 uint8_t *t4;
 uint8_t *const *t5;
 nav__6134_39 t6;
 bool t2;
 t0 = a0.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  return (nav__6134_39){(uint8_t *)((void const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),(uintptr_t)0ul};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = a0.ptr;
 t4 = t3;
 t5 = (uint8_t *const *)&t4;
 t0 = a0.len;
 t0 = t0 / (uintptr_t)1ul;
 t3 = (*t5);
 t3 = (uint8_t *)(((uintptr_t)t3) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t6.ptr = t3;
 t6.len = t0;
 return t6;
}

void c_instrument_hooks_deinit__232(struct instruments_root_InstrumentHooks__188 *const a0) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t4;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *t5;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t6;
 struct instruments_perf_PerfInstrument__200 *t8;
 bool t0;
 uint8_t t7;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t4 = (*t3);
  t5 = t4;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t5;
  t6 = (*t4);
  t7 = t6.tag;
  switch (t7) {
   case UINT8_C(0): {
    goto zig_block_1;
   }
   case UINT8_C(1): {
    t4 = (*t3);
    t8 = (struct instruments_perf_PerfInstrument__200 *)&t4->payload.perf;
    instruments_perf_PerfInstrument_deinit__338(t8);
    goto zig_block_1;
   }
   case UINT8_C(2): {
    goto zig_block_1;
   }
   default: zig_unreachable();
  }

  zig_block_1:;
  mem_Allocator_destroy__anon_3774__6135((struct mem_Allocator__206){((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)&heap_CAllocator_vtable__3476)}, t1);
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 return;
}

static void instruments_perf_PerfInstrument_deinit__338(struct instruments_perf_PerfInstrument__200 *const a0) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 struct fifo_UnixPipe_Reader__540 *t4;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 fifo_UnixPipe_Writer_deinit__992(t3);
 t2 = (*t1);
 t4 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 fifo_UnixPipe_Reader_deinit__999(t4);
 return;
}

static void mem_Allocator_destroy__anon_3774__6135(struct mem_Allocator__206 const a0, struct instruments_root_InstrumentHooks__188 *const a1) {
 struct mem_Allocator__206 const *t1;
 uint8_t *t2;
 uint8_t *t3;
 uint8_t *const *t4;
 struct mem_Allocator__206 t5;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t10;
 uint8_t (*t6)[72];
 nav__6135_52 t7;
 uintptr_t t9;
 struct mem_Allocator_VTable__209 const *const *t11;
 struct mem_Allocator_VTable__209 const *t12;
 void (*const *t13)(void *, nav__6135_52, uint8_t, uintptr_t);
 void (*t14)(void *, nav__6135_52, uint8_t, uintptr_t);
 void *t15;
 uint8_t t8;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = (uint8_t *)a1;
 t3 = t2;
 t4 = (uint8_t *const *)&t3;
 t5 = (*t1);
 t2 = (*t4);
 t2 = (uint8_t *)(((uintptr_t)t2) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t6 = (uint8_t (*)[72])t2;
 t7.ptr = &(*t6)[(uintptr_t)0ul];
 t7.len = (uintptr_t)72ul;
 t8 = mem_Alignment_fromByteUnits__628((uintptr_t)8ul);
 t9 = (uintptr_t)zig_return_address();
 t10 = t5;
 t1 = (struct mem_Allocator__206 const *)&t10;
 t11 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t12 = (*t11);
 t13 = (void (*const *)(void *, nav__6135_52, uint8_t, uintptr_t))&t12->free;
 t14 = (*t13);
 t15 = t5.ptr;
 t14(t15, t7, t8, t9);
 return;
}

static void fifo_UnixPipe_Writer_deinit__992(struct fifo_UnixPipe_Writer__538 *const a0) {
 struct fifo_UnixPipe_Writer__538 *const *t1;
 struct fifo_UnixPipe_Writer__538 *t2;
 struct fifo_UnixPipe_Writer__538 *t0;
 struct fs_File__546 *t3;
 struct fs_File__546 t4;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Writer__538 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fs_File__546 *)&t2->file;
 t4 = (*t3);
 fs_File_close__1090(t4);
 return;
}

static void fifo_UnixPipe_Reader_deinit__999(struct fifo_UnixPipe_Reader__540 *const a0) {
 struct fifo_UnixPipe_Reader__540 *const *t1;
 struct fifo_UnixPipe_Reader__540 *t2;
 struct fifo_UnixPipe_Reader__540 *t0;
 struct fs_File__546 *t3;
 struct fs_File__546 t4;
 t0 = a0;
 t1 = (struct fifo_UnixPipe_Reader__540 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fs_File__546 *)&t2->file;
 t4 = (*t3);
 fs_File_close__1090(t4);
 return;
}

static void fs_File_close__1090(struct fs_File__546 const a0) {
 int32_t t0;
 t0 = a0.handle;
 posix_close__1333(t0);
 return;
}

bool c_instrument_hooks_is_instrumented__233(struct instruments_root_InstrumentHooks__188 *const a0) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t4;
 struct instruments_perf_PerfInstrument__200 t8;
 struct instruments_perf_PerfInstrument__200 t9;
 bool t0;
 bool t6;
 bool t7;
 uint8_t t5;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t1 = (*t3);
  t4 = (*t1);
  t5 = t4.tag;
  switch (t5) {
   case UINT8_C(0): {
    t5 = running_on_valgrind();
    t7 = t5 > UINT8_C(0);
    t6 = t7;
    goto zig_block_2;
   }
   case UINT8_C(1): {
    t8 = t4.payload.perf;
    t9 = t8;
    t7 = instruments_perf_PerfInstrument_is_instrumented__340(&t9);
    t0 = t7;
    goto zig_block_1;
   }
   case UINT8_C(2): {
    t6 = false;
    goto zig_block_2;
   }
   default: zig_unreachable();
  }

  zig_block_2:;
  t0 = t6;
  goto zig_block_1;

  zig_block_1:;
  return t0;
 }
 goto zig_block_0;

 zig_block_0:;
 return false;
}

static zig_cold uint16_t instruments_perf_PerfInstrument_start_benchmark__341(struct instruments_perf_PerfInstrument__200 *const a0) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 struct fifo_UnixPipe_Reader__540 *t5;
 uint16_t t4;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 t4 = fifo_UnixPipe_Writer_sendCmd__991(t3, (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(1)});
 if (t4) {
  return t4;
 }
 t2 = (*t1);
 t5 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 t4 = fifo_UnixPipe_Reader_waitForAck__998(t5, (nav__341_66){UINT64_C(0xaaaaaaaaaaaaaaaa),true});
 if (t4) {
  return t4;
 }
 return 0;
}

uint8_t c_instrument_hooks_start_benchmark__234(struct instruments_root_InstrumentHooks__188 *const a0) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *t5;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t6;
 struct instruments_perf_PerfInstrument__200 *t8;
 uint16_t t4;
 uint16_t t9;
 uint16_t t10;
 bool t0;
 uint8_t t7;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t1 = (*t3);
  t5 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t5;
  t6 = (*t1);
  t7 = t6.tag;
  t0 = t7 == UINT8_C(1);
  if (t0) {
   t1 = (*t3);
   t8 = (struct instruments_perf_PerfInstrument__200 *)&t1->payload.perf;
   t9 = instruments_perf_PerfInstrument_start_benchmark__341(t8);
   memcpy(&t10, &t9, sizeof(uint16_t));
   t4 = t10;
   goto zig_block_2;
  }
  t6 = (*t1);
  t7 = t6.tag;
  t0 = t7 == UINT8_C(0);
  if (t0) {
   callgrind_zero_stats();
   callgrind_start_instrumentation();
   t4 = 0;
   goto zig_block_2;
  }
  goto zig_block_4;

  zig_block_4:;
  goto zig_block_3;

  zig_block_3:;
  t4 = 0;
  goto zig_block_2;

  zig_block_2:;
  memcpy(&t10, &t4, sizeof(uint16_t));
  t0 = t10 == UINT16_C(0);
  if (t0) {
   goto zig_block_1;
  }
  return UINT8_C(1);

  zig_block_1:;
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 return UINT8_C(0);
}

static zig_cold uint16_t instruments_perf_PerfInstrument_stop_benchmark__342(struct instruments_perf_PerfInstrument__200 *const a0) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 struct fifo_UnixPipe_Reader__540 *t5;
 uint16_t t4;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 t4 = fifo_UnixPipe_Writer_sendCmd__991(t3, (struct shared_Command__1919){{{{(uint8_t const *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},UINT32_C(0xaaaaaaaa)}},UINT8_C(2)});
 if (t4) {
  return t4;
 }
 t2 = (*t1);
 t5 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 t4 = fifo_UnixPipe_Reader_waitForAck__998(t5, (nav__342_66){UINT64_C(0xaaaaaaaaaaaaaaaa),true});
 if (t4) {
  return t4;
 }
 return 0;
}

uint8_t c_instrument_hooks_stop_benchmark__235(struct instruments_root_InstrumentHooks__188 *const a0) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *t5;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t6;
 struct instruments_perf_PerfInstrument__200 *t8;
 uint16_t t4;
 uint16_t t9;
 uint16_t t10;
 bool t0;
 uint8_t t7;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t1 = (*t3);
  t5 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t5;
  t6 = (*t1);
  t7 = t6.tag;
  t0 = t7 == UINT8_C(0);
  if (t0) {
   callgrind_stop_instrumentation();
   t4 = 0;
   goto zig_block_2;
  }
  t6 = (*t1);
  t7 = t6.tag;
  t0 = t7 == UINT8_C(1);
  if (t0) {
   t1 = (*t3);
   t8 = (struct instruments_perf_PerfInstrument__200 *)&t1->payload.perf;
   t9 = instruments_perf_PerfInstrument_stop_benchmark__342(t8);
   memcpy(&t10, &t9, sizeof(uint16_t));
   t4 = t10;
   goto zig_block_2;
  }
  goto zig_block_4;

  zig_block_4:;
  goto zig_block_3;

  zig_block_3:;
  t4 = 0;
  goto zig_block_2;

  zig_block_2:;
  memcpy(&t10, &t4, sizeof(uint16_t));
  t0 = t10 == UINT16_C(0);
  if (t0) {
   goto zig_block_1;
  }
  return UINT8_C(1);

  zig_block_1:;
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 return UINT8_C(0);
}

static uint16_t instruments_perf_PerfInstrument_set_executed_benchmark__343(struct instruments_perf_PerfInstrument__200 *const a0, uint32_t const a1, uint8_t const *const a2) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 nav__343_56 t4;
 nav__343_56 t5;
 struct shared_Command__struct_1922__1922 t6;
 struct shared_Command__1919 t7;
 struct fifo_UnixPipe_Reader__540 *t9;
 uint16_t t8;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 t4 = mem_span__anon_3885__6140(a2);
 memcpy(&t5, &t4, sizeof(nav__343_56));
 t6.uri = t5;
 t6.pid = a1;
 t7.tag = UINT8_C(0);
 t7.payload.ExecutedBenchmark = t6;
 t8 = fifo_UnixPipe_Writer_sendCmd__991(t3, t7);
 if (t8) {
  return t8;
 }
 t2 = (*t1);
 t9 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 t8 = fifo_UnixPipe_Reader_waitForAck__998(t9, (nav__343_66){UINT64_C(0xaaaaaaaaaaaaaaaa),true});
 if (t8) {
  return t8;
 }
 return 0;
}

uint8_t c_instrument_hooks_executed_benchmark__236(struct instruments_root_InstrumentHooks__188 *const a0, uint32_t const a1, uint8_t const *const a2) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *t5;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t6;
 uint8_t const *t8;
 struct instruments_perf_PerfInstrument__200 *t9;
 uint16_t t4;
 uint16_t t10;
 bool t0;
 uint8_t t7;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t1 = (*t3);
  t5 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t5;
  t6 = (*t1);
  t7 = t6.tag;
  switch (t7) {
   case UINT8_C(0): {
    t8 = (uint8_t const *)a2;
    callgrind_dump_stats_at(t8);
    goto zig_block_3;
   }
   case UINT8_C(1): {
    t1 = (*t3);
    t9 = (struct instruments_perf_PerfInstrument__200 *)&t1->payload.perf;
    t10 = instruments_perf_PerfInstrument_set_executed_benchmark__343(t9, a1, a2);
    if (t10) {
     t4 = t10;
     goto zig_block_2;
    }
    goto zig_block_3;
   }
   case UINT8_C(2): {
    goto zig_block_3;
   }
   default: zig_unreachable();
  }

  zig_block_3:;
  t4 = 0;
  goto zig_block_2;

  zig_block_2:;
  memcpy(&t10, &t4, sizeof(uint16_t));
  t0 = t10 == UINT16_C(0);
  if (t0) {
   goto zig_block_1;
  }
  return UINT8_C(1);

  zig_block_1:;
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 return UINT8_C(0);
}

static nav__6140_39 mem_span__anon_3885__6140(uint8_t const *const a0) {
 uint8_t const *const *t1;
 uintptr_t t2;
 uint8_t const *t3;
 uint8_t const *t0;
 nav__6140_39 t4;
 t0 = a0;
 t1 = (uint8_t const *const *)&t0;
 t2 = mem_len__anon_3894__6141(a0);
 t3 = (*t1);
 t3 = (uint8_t const *)(((uintptr_t)t3) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t4.ptr = t3;
 t4.len = t2;
 return t4;
}

static uintptr_t mem_len__anon_3894__6141(uint8_t const *const a0) {
 uint8_t const *t1;
 uintptr_t t2;
 bool t0;
 t0 = a0 != NULL;
 debug_assert__179(t0);
 t1 = (uint8_t const *)a0;
 t2 = mem_indexOfSentinel__anon_3902__6142(t1);
 return t2;
}

static uintptr_t mem_indexOfSentinel__anon_3902__6142(uint8_t const *const a0) {
 static uint8_t const t11[32] = "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
 static uint8_t const t18[32] = "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000";
 uint8_t const *const *t1;
 uintptr_t t4;
 uintptr_t t6;
 uintptr_t t16;
 uintptr_t t2;
 uint8_t const *t5;
 uint8_t const *t0;
 uint64_t t7;
 uint8_t const (*t8)[32];
 uint8_t const (*t17)[32];
 bool t3;
 uint8_t t9[32];
 uint8_t t10[32];
 bool t12[32];
 nav__6142_45 t13;
 nav__6142_47 t14;
 uint8_t t15;
 t0 = a0;
 t1 = (uint8_t const *const *)&t0;
 t2 = (uintptr_t)0ul;
 t3 = math_isPowerOfTwo__anon_3913__6143();
 if (t3) {
  debug_assert__179(true);
  t4 = t2;
  t5 = (*t1);
  t5 = (uint8_t const *)&t5[t4];
  t4 = (uintptr_t)t5;
  t6 = t4 & (uintptr_t)4095ul;
  t7 = t6;
  t3 = t7 <= UINT64_C(4064);
  if (t3) {
   t6 = t2;
   t5 = (*t1);
   t5 = (uint8_t const *)(((uintptr_t)t5) + (t6*sizeof(uint8_t)));
   t8 = (uint8_t const (*)[32])t5;
   memcpy(t9, (const char *)t8, sizeof(uint8_t[32]));
   memcpy(&t10, &t9, sizeof(uint8_t[32]));
   for (t6 = (uintptr_t)0ul; t6 < (uintptr_t)32ul; t6 += (uintptr_t)1ul) {
    t12[t6] = t10[t6] == t11[t6];
   }
   t3 = false;
   for (t6 = (uintptr_t)0ul; t6 < (uintptr_t)32ul; t6 += (uintptr_t)1ul) {
    t3 |= t12[t6];
   }
   if (t3) {
    t4 = t2;
    memcpy(t13.array, t12, sizeof(nav__6142_45));
    t14 = simd_firstTrue__anon_4125__6508(t13);
    t15 = t14.payload;
    t6 = (uintptr_t)t15;
    t6 = t4 + t6;
    return t6;
   }
   goto zig_block_2;

   zig_block_2:;
   t6 = t2;
   t16 = mem_alignForward__anon_4128__6509(t4, (uintptr_t)32ul);
   t4 = t16 - t4;
   t4 = t4 / (uintptr_t)1ul;
   t4 = t6 + t4;
   t2 = t4;
   goto zig_block_1;
  }
  zig_loop_79:
  t4 = t2;
  t5 = (*t1);
  t5 = (uint8_t const *)&t5[t4];
  t4 = (uintptr_t)t5;
  t4 = t4 & (uintptr_t)31ul;
  t7 = t4;
  t3 = t7 != UINT64_C(0);
  if (t3) {
   t4 = t2;
   t15 = a0[t4];
   t3 = t15 == UINT8_C(0);
   if (t3) {
    t4 = t2;
    return t4;
   }
   goto zig_block_5;

   zig_block_5:;
   t4 = t2;
   t4 = t4 + (uintptr_t)1ul;
   t2 = t4;
   goto zig_block_4;
  }
  goto zig_block_3;

  zig_block_4:;
  goto zig_loop_79;

  zig_block_3:;
  goto zig_block_1;

  zig_block_1:;
  t6 = t2;
  t5 = (*t1);
  t5 = (uint8_t const *)&t5[t6];
  t6 = (uintptr_t)t5;
  t3 = mem_isAligned__499(t6, (uintptr_t)32ul);
  debug_assert__179(t3);
  zig_loop_114:
  t6 = t2;
  t5 = (*t1);
  t5 = (uint8_t const *)(((uintptr_t)t5) + (t6*sizeof(uint8_t)));
  t8 = (uint8_t const (*)[32])t5;
  t17 = (uint8_t const (*)[32])t8;
  memcpy(t10, (const char *)t17, sizeof(uint8_t[32]));
  for (t6 = (uintptr_t)0ul; t6 < (uintptr_t)32ul; t6 += (uintptr_t)1ul) {
   t12[t6] = t10[t6] == t18[t6];
  }
  t3 = false;
  for (t6 = (uintptr_t)0ul; t6 < (uintptr_t)32ul; t6 += (uintptr_t)1ul) {
   t3 |= t12[t6];
  }
  if (t3) {
   t6 = t2;
   memcpy(t13.array, t12, sizeof(nav__6142_45));
   t14 = simd_firstTrue__anon_4125__6508(t13);
   t15 = t14.payload;
   t4 = (uintptr_t)t15;
   t4 = t6 + t4;
   return t4;
  }
  goto zig_block_6;

  zig_block_6:;
  t6 = t2;
  t6 = t6 + (uintptr_t)32ul;
  t2 = t6;
  goto zig_loop_114;
 }
 goto zig_block_0;

 zig_block_0:;
 zig_loop_145:
 t16 = t2;
 t15 = a0[t16];
 t3 = t15 != UINT8_C(0);
 if (t3) {
  t16 = t2;
  t16 = t16 + (uintptr_t)1ul;
  t2 = t16;
  goto zig_block_8;
 }
 goto zig_block_7;

 zig_block_8:;
 goto zig_loop_145;

 zig_block_7:;
 t16 = t2;
 return t16;
}

static bool math_isPowerOfTwo__anon_3913__6143(void) {
 debug_assert__179(true);
 return true;
}

static nav__6508_38 simd_firstTrue__anon_4125__6508(nav__6508_40 const a0) {
 static uint8_t const t2[32] = {UINT8_C(0),UINT8_C(1),UINT8_C(2),UINT8_C(3),UINT8_C(4),UINT8_C(5),UINT8_C(6),UINT8_C(7),UINT8_C(8),UINT8_C(9),UINT8_C(10),UINT8_C(11),UINT8_C(12),UINT8_C(13),UINT8_C(14),UINT8_C(15),UINT8_C(16),UINT8_C(17),UINT8_C(18),UINT8_C(19),UINT8_C(20),UINT8_C(21),UINT8_C(22),UINT8_C(23),UINT8_C(24),UINT8_C(25),UINT8_C(26),UINT8_C(27),UINT8_C(28),UINT8_C(29),UINT8_C(30),UINT8_C(31)};
 static uint8_t const t3[32] = {UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31),UINT8_C(31)};
 uintptr_t t1;
 bool t0;
 uint8_t t4[32];
 uint8_t t5;
 nav__6508_38 t6;
 t0 = false;
 for (t1 = (uintptr_t)0ul; t1 < (uintptr_t)32ul; t1 += (uintptr_t)1ul) {
  t0 |= a0.array[t1];
 }
 t0 = !t0;
 if (t0) {
  return (nav__6508_38){true,UINT8_C(0xa)};
 }
 goto zig_block_0;

 zig_block_0:;
 for (t1 = (uintptr_t)0ul; t1 < (uintptr_t)32ul; t1 += (uintptr_t)1ul) {
  t4[t1] = a0.array[t1] ? t2[t1] : t3[t1];
 }
 t5 = UINT8_C(31);
 for (t1 = (uintptr_t)0ul; t1 < (uintptr_t)32ul; t1 += (uintptr_t)1ul) {
  t5 = t5 < t4[t1] ? t5 : t4[t1];
 }
 t6.is_null = false;
 t6.payload = t5;
 return t6;
}

static uintptr_t mem_alignForward__anon_4128__6509(uintptr_t const a0, uintptr_t const a1) {
 uintptr_t t1;
 bool t0;
 t0 = mem_isValidAlignGeneric__anon_4186__6510(a1);
 debug_assert__179(t0);
 t1 = a1 - (uintptr_t)1ul;
 t1 = a0 + t1;
 t1 = mem_alignBackward__anon_4187__6511(t1, a1);
 return t1;
}

static bool mem_isAligned__499(uintptr_t const a0, uintptr_t const a1) {
 uint64_t t0;
 uint64_t t1;
 bool t2;
 t0 = a0;
 t1 = a1;
 t2 = mem_isAlignedGeneric__anon_4193__6512(t0, t1);
 return t2;
}

static bool mem_isValidAlignGeneric__anon_4186__6510(uintptr_t const a0) {
 uint64_t t0;
 bool t1;
 bool t2;
 t0 = a0;
 t1 = t0 > UINT64_C(0);
 if (t1) {
  t1 = math_isPowerOfTwo__anon_2984__5884(a0);
  t2 = t1;
  goto zig_block_0;
 }
 t2 = false;
 goto zig_block_0;

 zig_block_0:;
 return t2;
}

static uintptr_t mem_alignBackward__anon_4187__6511(uintptr_t const a0, uintptr_t const a1) {
 uintptr_t t1;
 bool t0;
 t0 = mem_isValidAlignGeneric__anon_4186__6510(a1);
 debug_assert__179(t0);
 t1 = a1 - (uintptr_t)1ul;
 t1 = zig_not_u64(t1, UINT8_C(64));
 t1 = a0 & t1;
 return t1;
}

static bool mem_isAlignedGeneric__anon_4193__6512(uint64_t const a0, uint64_t const a1) {
 uint64_t t0;
 bool t1;
 t0 = mem_alignBackward__anon_4195__6513(a0, a1);
 t1 = t0 == a0;
 return t1;
}

static uint64_t mem_alignBackward__anon_4195__6513(uint64_t const a0, uint64_t const a1) {
 uint64_t t1;
 bool t0;
 t0 = mem_isValidAlignGeneric__anon_4197__6514(a1);
 debug_assert__179(t0);
 t1 = a1 - UINT64_C(1);
 t1 = zig_not_u64(t1, UINT8_C(64));
 t1 = a0 & t1;
 return t1;
}

static bool mem_isValidAlignGeneric__anon_4197__6514(uint64_t const a0) {
 bool t0;
 bool t1;
 t0 = a0 > UINT64_C(0);
 if (t0) {
  t0 = math_isPowerOfTwo__anon_4198__6515(a0);
  t1 = t0;
  goto zig_block_0;
 }
 t1 = false;
 goto zig_block_0;

 zig_block_0:;
 return t1;
}

static bool math_isPowerOfTwo__anon_4198__6515(uint64_t const a0) {
 uint64_t t1;
 bool t0;
 t0 = a0 > UINT64_C(0);
 debug_assert__179(t0);
 t1 = a0 - UINT64_C(1);
 t1 = a0 & t1;
 t0 = t1 == UINT64_C(0);
 return t0;
}

static uint16_t instruments_perf_PerfInstrument_set_integration__344(struct instruments_perf_PerfInstrument__200 *const a0, uint8_t const *const a1, uint8_t const *const a2) {
 struct instruments_perf_PerfInstrument__200 *const *t1;
 struct instruments_perf_PerfInstrument__200 *t2;
 struct instruments_perf_PerfInstrument__200 *t0;
 struct fifo_UnixPipe_Writer__538 *t3;
 nav__344_56 t4;
 nav__344_56 t5;
 nav__344_56 t6;
 struct shared_Command__struct_1923__1923 t7;
 struct shared_Command__1919 t8;
 struct fifo_UnixPipe_Reader__540 *t10;
 uint16_t t9;
 t0 = a0;
 t1 = (struct instruments_perf_PerfInstrument__200 *const *)&t0;
 t2 = (*t1);
 t3 = (struct fifo_UnixPipe_Writer__538 *)&t2->writer;
 t4 = mem_span__anon_3885__6140(a1);
 t5 = mem_span__anon_3885__6140(a2);
 memcpy(&t6, &t4, sizeof(nav__344_56));
 memcpy(&t4, &t5, sizeof(nav__344_56));
 t7.name = t6;
 t7.version = t4;
 t8.tag = UINT8_C(5);
 t8.payload.SetIntegration = t7;
 t9 = fifo_UnixPipe_Writer_sendCmd__991(t3, t8);
 if (t9) {
  return t9;
 }
 t2 = (*t1);
 t10 = (struct fifo_UnixPipe_Reader__540 *)&t2->reader;
 t9 = fifo_UnixPipe_Reader_waitForAck__998(t10, (nav__344_66){UINT64_C(0xaaaaaaaaaaaaaaaa),true});
 if (t9) {
  return t9;
 }
 return 0;
}

uint8_t c_instrument_hooks_set_integration__237(struct instruments_root_InstrumentHooks__188 *const a0, uint8_t const *const a1, uint8_t const *const a2) {
 struct instruments_root_InstrumentHooks__188 *t1;
 struct instruments_root_InstrumentHooks__188 *t2;
 struct instruments_root_InstrumentHooks__188 *t5;
 struct instruments_root_InstrumentHooks__188 *const *t3;
 struct instruments_root_InstrumentHooks__188 t6;
 struct instruments_valgrind_ValgrindInstrument__195 *t8;
 struct instruments_valgrind_ValgrindInstrument__195 t9;
 struct instruments_valgrind_ValgrindInstrument__195 t11;
 struct instruments_valgrind_ValgrindInstrument__195 const *t12;
 struct mem_Allocator__206 t13;
 nav__237_62 t14;
 nav__237_67 t15;
 nav__237_65 t17;
 uint8_t *t18;
 uint8_t const *t19;
 struct mem_Allocator__206 const *t20;
 struct instruments_perf_PerfInstrument__200 *t21;
 uint16_t t4;
 uint16_t t10;
 uint16_t t16;
 bool t0;
 uint8_t t7;
 t0 = a0 != NULL;
 if (t0) {
  t1 = a0;
  t2 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t2;
  t1 = (*t3);
  t5 = t1;
  t3 = (struct instruments_root_InstrumentHooks__188 *const *)&t5;
  t6 = (*t1);
  t7 = t6.tag;
  switch (t7) {
   case UINT8_C(0): {
    t1 = (*t3);
    t8 = (struct instruments_valgrind_ValgrindInstrument__195 *)&t1->payload.valgrind;
    t9 = (*t8);
    t11 = t9;
    t12 = (struct instruments_valgrind_ValgrindInstrument__195 const *)&t11;
    t13 = t9.allocator;
    t14.f0 = a1;
    t14.f1 = a2;
    t15 = fmt_allocPrintZ__anon_4232__6516(t13, t14);
    if (t15.error) {
     t16 = t15.error;
     t10 = t16;
     goto zig_block_4;
    }
    t17 = t15.payload;
    t18 = t17.ptr;
    t19 = (uint8_t const *)t18;
    callgrind_dump_stats_at(t19);
    t20 = (struct mem_Allocator__206 const *)&t12->allocator;
    t13 = (*t20);
    mem_Allocator_free__anon_4234__6517(t13, t17);
    t10 = 0;
    goto zig_block_4;

    zig_block_4:;
    memcpy(&t16, &t10, sizeof(uint16_t));
    if (t16) {
     t4 = t16;
     goto zig_block_2;
    }
    goto zig_block_3;
   }
   case UINT8_C(1): {
    t1 = (*t3);
    t21 = (struct instruments_perf_PerfInstrument__200 *)&t1->payload.perf;
    t16 = instruments_perf_PerfInstrument_set_integration__344(t21, a1, a2);
    if (t16) {
     t4 = t16;
     goto zig_block_2;
    }
    goto zig_block_3;
   }
   case UINT8_C(2): {
    goto zig_block_3;
   }
   default: zig_unreachable();
  }

  zig_block_3:;
  t4 = 0;
  goto zig_block_2;

  zig_block_2:;
  memcpy(&t16, &t4, sizeof(uint16_t));
  t0 = t16 == UINT16_C(0);
  if (t0) {
   goto zig_block_1;
  }
  return UINT8_C(1);

  zig_block_1:;
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 return UINT8_C(0);
}

static nav__6516_40 fmt_allocPrintZ__anon_4232__6516(struct mem_Allocator__206 const a0, nav__6516_43 const a1) {
 nav__6516_40 t0;
 nav__6516_39 t2;
 nav__6516_39 t3;
 nav__6516_39 const *t4;
 uintptr_t t5;
 uint8_t *t6;
 uint16_t t1;
 t0 = fmt_allocPrint__anon_4259__6518(a0, a1);
 if (t0.error) {
  t1 = t0.error;
  t0.payload = (nav__6516_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t0.error = t1;
  return t0;
 }
 t2 = t0.payload;
 t3 = t2;
 t4 = (nav__6516_39 const *)&t3;
 t5 = t2.len;
 t5 = t5 - (uintptr_t)1ul;
 t2 = (*t4);
 t6 = t2.ptr;
 t6 = (uint8_t *)(((uintptr_t)t6) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t2.ptr = t6;
 t2.len = t5;
 t0.payload = t2;
 t0.error = UINT16_C(0);
 return t0;
}

static void mem_Allocator_free__anon_4234__6517(struct mem_Allocator__206 const a0, nav__6517_40 const a1) {
 struct mem_Allocator__206 const *t1;
 nav__6517_40 t2;
 uintptr_t t3;
 uint64_t t4;
 uint8_t *t6;
 uint8_t *t7;
 uint8_t *t8;
 uint8_t *const *t9;
 struct mem_Allocator__206 t10;
 struct mem_Allocator__206 t0;
 struct mem_Allocator__206 t12;
 struct mem_Allocator_VTable__209 const *const *t13;
 struct mem_Allocator_VTable__209 const *t14;
 void (*const *t15)(void *, nav__6517_40, uint8_t, uintptr_t);
 void (*t16)(void *, nav__6517_40, uint8_t, uintptr_t);
 void *t17;
 bool t5;
 uint8_t t11;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t2 = mem_sliceAsBytes__anon_4266__6519(a1);
 t3 = t2.len;
 t3 = t3 + (uintptr_t)1ul;
 t4 = t3;
 t5 = t4 == UINT64_C(0);
 if (t5) {
  return;
 }
 goto zig_block_0;

 zig_block_0:;
 t6 = t2.ptr;
 t7 = (uint8_t *)t6;
 t8 = t7;
 t9 = (uint8_t *const *)&t8;
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t2.ptr = t7;
 t2.len = t3;
 t10 = (*t1);
 t7 = (*t9);
 t7 = (uint8_t *)(((uintptr_t)t7) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t2.ptr = t7;
 t2.len = t3;
 t11 = mem_Alignment_fromByteUnits__628((uintptr_t)1ul);
 t3 = (uintptr_t)zig_return_address();
 t12 = t10;
 t1 = (struct mem_Allocator__206 const *)&t12;
 t13 = (struct mem_Allocator_VTable__209 const *const *)&t1->vtable;
 t14 = (*t13);
 t15 = (void (*const *)(void *, nav__6517_40, uint8_t, uintptr_t))&t14->free;
 t16 = (*t15);
 t17 = t10.ptr;
 t16(t17, t2, t11, t3);
 return;
}

static nav__6518_40 fmt_allocPrint__anon_4259__6518(struct mem_Allocator__206 const a0, nav__6518_43 const a1) {
 struct mem_Allocator__206 const *t1;
 uintptr_t t2;
 uintptr_t t6;
 uint64_t t3;
 nav__6518_54 t4;
 struct mem_Allocator__206 t7;
 struct mem_Allocator__206 t0;
 nav__6518_40 t8;
 nav__6518_40 t11;
 nav__6518_39 t10;
 uint16_t t9;
 bool t5;
 t0 = a0;
 t1 = (struct mem_Allocator__206 const *)&t0;
 t3 = fmt_count__anon_4273__6520(a1);
 t4 = math_cast__anon_4275__6521(t3);
 t5 = t4.is_null != true;
 if (t5) {
  t6 = t4.payload;
  t2 = t6;
  goto zig_block_0;
 }
 return (nav__6518_40){{(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},zig_error_OutOfMemory};

 zig_block_0:;
 t7 = (*t1);
 t8 = mem_Allocator_alloc__anon_2114__4100(t7, t2);
 if (t8.error) {
  t9 = t8.error;
  t8.payload = (nav__6518_39){(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul};
  t8.error = t9;
  return t8;
 }
 t10 = t8.payload;
 t8 = fmt_bufPrint__anon_4284__6522(t10, a1);
 t5 = t8.error == UINT16_C(0);
 if (t5) {
  t10 = t8.payload;
  t8.payload = t10;
  t8.error = UINT16_C(0);
  t11 = t8;
  goto zig_block_1;
 }
 t9 = t8.error;
 switch (t9) {
  case zig_error_NoSpaceLeft: {
   zig_unreachable();
  }
  default: zig_unreachable();
 }

 zig_block_1:;
 return t11;
}

static nav__6519_39 mem_sliceAsBytes__anon_4266__6519(nav__6519_39 const a0) {
 uintptr_t t0;
 uint64_t t1;
 uint8_t *t3;
 uint8_t *t4;
 uint8_t *t5;
 uint8_t *const *t6;
 nav__6519_39 t7;
 bool t2;
 t0 = a0.len;
 t1 = t0;
 t2 = t1 == UINT64_C(0);
 if (t2) {
  goto zig_block_0;
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = a0.ptr;
 t4 = (uint8_t *)t3;
 t5 = t4;
 t6 = (uint8_t *const *)&t5;
 t0 = a0.len;
 t4 = (*t6);
 t4 = (uint8_t *)(((uintptr_t)t4) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t7.ptr = t4;
 t7.len = t0;
 return t7;
}

static uint16_t fmt_format__anon_4341__6561(struct io_Writer__3628 const a0, nav__6561_40 const a1) {
 struct io_Writer__3628 const *t1;
 struct io_Writer__3628 t2;
 struct io_Writer__3628 t0;
 uint8_t const *t4;
 uint16_t t3;
 t0 = a0;
 t1 = (struct io_Writer__3628 const *)&t0;
 t2 = (*t1);
 t3 = io_Writer_writeAll__6036(t2, (nav__6561_44){(uint8_t const *)&__anon_4395,(uintptr_t)10ul});
 if (t3) {
  return t3;
 }
 t4 = a1.f0;
 t3 = fmt_formatType__anon_4668__6729(t4, (struct fmt_FormatOptions__4654){{(uintptr_t)0xaaaaaaaaaaaaaaaaul,true},{(uintptr_t)0xaaaaaaaaaaaaaaaaul,true},UINT32_C(32),UINT8_C(2)}, a0, (uintptr_t)3ul);
 if (t3) {
  return t3;
 }
 t2 = (*t1);
 t3 = io_Writer_writeAll__6036(t2, (nav__6561_44){(uint8_t const *)&__anon_4676,(uintptr_t)1ul});
 if (t3) {
  return t3;
 }
 t4 = a1.f1;
 t3 = fmt_formatType__anon_4668__6729(t4, (struct fmt_FormatOptions__4654){{(uintptr_t)0xaaaaaaaaaaaaaaaaul,true},{(uintptr_t)0xaaaaaaaaaaaaaaaaul,true},UINT32_C(32),UINT8_C(2)}, a0, (uintptr_t)3ul);
 if (t3) {
  return t3;
 }
 t2 = (*t1);
 t3 = io_Writer_writeAll__6036(t2, (nav__6561_44){(uint8_t const *)&__anon_4696,(uintptr_t)1ul});
 if (t3) {
  return t3;
 }
 return 0;
}

static uint64_t fmt_count__anon_4273__6520(nav__6520_39 const a0) {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 t1;
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 t0;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 t2;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 t3;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 const *t4;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 const *t5;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 const *const *t6;
 void const **t8;
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *t9;
 void const *t10;
 nav__6520_53 (**t11)(void const *, nav__6520_54);
 struct io_Writer__3628 t12;
 struct io_Writer__3628 t7;
 uint64_t t15;
 uint16_t t13;
 bool t14;
 t1 = io_counting_writer_countingWriter__anon_4305__6547();
 t0 = t1;
 t2 = io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_writer__6546(&t0);
 t3 = t2;
 t4 = (struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 const *)&t3;
 t5 = t4;
 t6 = (struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 const *const *)&t5;
 t8 = (void const **)&t7.context;
 t4 = (*t6);
 t9 = (struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *)&t4->context;
 t10 = (void const *)t9;
 (*t8) = t10;
 t11 = (nav__6520_53 (**)(void const *, nav__6520_54))&t7.writeFn;
 (*t11) = &io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWr__6560;
 t12 = t7;
 t13 = fmt_format__anon_4341__6561(t12, a0);
 t14 = t13 == UINT16_C(0);
 if (t14) {
  goto zig_block_0;
 }
 zig_unreachable();

 zig_block_0:;
 t1 = t0;
 t15 = t1.bytes_written;
 return t15;
}

static nav__6521_38 math_cast__anon_4275__6521(uint64_t const a0) {
 uintptr_t t0;
 nav__6521_38 t1;
 t0 = a0;
 t1.is_null = false;
 t1.payload = t0;
 return t1;
}

static nav__6522_40 fmt_bufPrint__anon_4284__6522(nav__6522_39 const a0, nav__6522_42 const a1) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t1;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t0;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 t2;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 t3;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 const *t4;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 const *t5;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 const *const *t6;
 void const **t8;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t9;
 void const *t10;
 nav__6522_57 (**t11)(void const *, nav__6522_58);
 struct io_Writer__3628 t12;
 struct io_Writer__3628 t7;
 nav__6522_39 t15;
 nav__6522_40 t16;
 uint16_t t13;
 bool t14;
 t1 = io_fixed_buffer_stream_fixedBufferStream__anon_2136__4132(a0);
 t0 = t1;
 t2 = io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_writer__4122(&t0);
 t3 = t2;
 t4 = (struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 const *)&t3;
 t5 = t4;
 t6 = (struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 const *const *)&t5;
 t8 = (void const **)&t7.context;
 t4 = (*t6);
 t9 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t4->context;
 t10 = (void const *)t9;
 (*t8) = t10;
 t11 = (nav__6522_57 (**)(void const *, nav__6522_58))&t7.writeFn;
 (*t11) = &io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write__6742;
 t12 = t7;
 t13 = fmt_format__anon_4341__6561(t12, a1);
 t14 = t13 == UINT16_C(0);
 if (t14) {
  goto zig_block_0;
 }
 switch (t13) {
  case zig_error_NoSpaceLeft: {
   return (nav__6522_40){{(uint8_t *)(uintptr_t)0xaaaaaaaaaaaaaaaaul, (uintptr_t)0xaaaaaaaaaaaaaaaaul},zig_error_NoSpaceLeft};
  }
  default: {
   zig_unreachable();
  }
 }

 zig_block_0:;
 t1 = t0;
 t15 = io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_getWritten__4130(t1);
 t16.payload = t15;
 t16.error = UINT16_C(0);
 return t16;
}

static struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 io_counting_writer_countingWriter__anon_4305__6547(void) {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 t0;
 t0 = (struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302){UINT64_C(0)};
 return t0;
}

static struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_writer__6546(struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const a0) {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 **t1;
 struct io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_2cerror_7b_7d_2c_28function_20_27write_27_29_29__4322 t0;
 t1 = (struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 **)&t0.context;
 (*t1) = a0;
 return t0;
}

static nav__6560_38 io_GenericWriter_28_2aio_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWr__6560(void const *const a0, nav__6560_41 const a1) {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *t0;
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *t1;
 nav__6560_38 t2;
 nav__6560_38 t3;
 t0 = (struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *)a0;
 t1 = (*t0);
 t2 = io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_write__6545(t1, a1);
 memcpy(&t3, &t2, sizeof(nav__6560_38));
 return t3;
}

static uint16_t fmt_formatType__anon_4668__6729(uint8_t const *const a0, struct fmt_FormatOptions__4654 const a1, struct io_Writer__3628 const a2, uintptr_t const a3) {
 nav__6729_47 t2;
 nav__6729_47 t3;
 uint8_t const *t0;
 struct io_Writer__3628 t1;
 uint16_t t4;
 uint16_t t5;
 (void)a3;
 t0 = a0;
 t1 = a2;
 t2 = mem_span__anon_3885__6140(a0);
 memcpy(&t3, &t2, sizeof(nav__6729_47));
 t4 = fmt_formatBuf__anon_4769__6743(t3, a1, a2);
 memcpy(&t5, &t4, sizeof(uint16_t));
 return t5;
}

static struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_writer__4122(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const a0) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 **t1;
 struct io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write_27_29_29__4712 t0;
 t1 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 **)&t0.context;
 (*t1) = a0;
 return t0;
}

static nav__6742_38 io_GenericWriter_28_2aio_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_2cerror_7bNoSpaceLeft_7d_2c_28function_20_27write__6742(void const *const a0, nav__6742_41 const a1) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t0;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t1;
 nav__6742_38 t2;
 nav__6742_38 t3;
 t0 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)a0;
 t1 = (*t0);
 t2 = io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_write__4125(t1, a1);
 memcpy(&t3, &t2, sizeof(nav__6742_38));
 return t3;
}

static nav__4130_39 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_getWritten__4130(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 const a0) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 const *t1;
 nav__4130_39 const *t2;
 uintptr_t t3;
 nav__4130_39 t4;
 uint8_t *t5;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 t0;
 t0 = a0;
 t1 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 const *)&t0;
 t2 = (nav__4130_39 const *)&t1->buffer;
 t3 = a0.pos;
 t4 = (*t2);
 t5 = t4.ptr;
 t5 = (uint8_t *)(((uintptr_t)t5) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t4.ptr = t5;
 t4.len = t3;
 return t4;
}

static nav__6545_38 io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29_write__6545(struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const a0, nav__6545_42 const a1) {
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *t1;
 nav__6545_38 t2;
 uintptr_t t3;
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *t4;
 struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *t0;
 uint64_t *t5;
 uint64_t t6;
 uint64_t t7;
 t0 = a0;
 t1 = (struct io_counting_writer_CountingWriter_28io_GenericWriter_28void_2cerror_7b_7d_2c_28function_20_27dummyWrite_27_29_29_29__4302 *const *)&t0;
 t2 = io_dummyWrite__4035(a1);
 t3 = t2.payload;
 t4 = (*t1);
 t5 = (uint64_t *)&t4->bytes_written;
 t6 = (*t5);
 t7 = t3;
 t7 = t6 + t7;
 (*t5) = t7;
 t2.payload = t3;
 t2.error = UINT16_C(0);
 return t2;
}

static nav__6573_38 unicode_utf8ByteSequenceLength__6573(uint8_t const a0) {
 nav__6573_38 t0;
 switch (a0) {
  default: if ((a0 >= UINT8_C(0) && a0 <= UINT8_C(127))) {
   t0 = (nav__6573_38){0,UINT8_C(1)};
   goto zig_block_0;
  }if ((a0 >= UINT8_C(192) && a0 <= UINT8_C(223))) {
   t0 = (nav__6573_38){0,UINT8_C(2)};
   goto zig_block_0;
  }if ((a0 >= UINT8_C(224) && a0 <= UINT8_C(239))) {
   t0 = (nav__6573_38){0,UINT8_C(3)};
   goto zig_block_0;
  }if ((a0 >= UINT8_C(240) && a0 <= UINT8_C(247))) {
   t0 = (nav__6573_38){0,UINT8_C(4)};
   goto zig_block_0;
  }{
   t0 = (nav__6573_38){zig_error_Utf8InvalidStartByte,UINT8_C(0x2)};
   goto zig_block_0;
  }
 }

 zig_block_0:;
 return t0;
}

static nav__6589_38 unicode_utf8CountCodepoints__6589(nav__6589_40 const a0) {
 nav__6589_40 const *t1;
 uintptr_t t4;
 uintptr_t t5;
 uintptr_t t2;
 uintptr_t t3;
 uint64_t t6;
 uint64_t t7;
 nav__6589_40 t9;
 nav__6589_40 t0;
 uint8_t const *t10;
 uint8_t const (*t11)[8];
 nav__6589_38 t16;
 nav__6589_50 t17;
 nav__6589_48 t14;
 uint16_t t15;
 bool t8;
 uint8_t t12[8];
 uint8_t t13;
 t0 = a0;
 t1 = (nav__6589_40 const *)&t0;
 t2 = (uintptr_t)0ul;
 t3 = (uintptr_t)0ul;
 zig_loop_9:
 t4 = t3;
 t5 = a0.len;
 t6 = t4;
 t7 = t5;
 t8 = t6 < t7;
 if (t8) {
  zig_loop_18:
  t5 = t3;
  t5 = t5 + (uintptr_t)8ul;
  t4 = a0.len;
  t7 = t5;
  t6 = t4;
  t8 = t7 <= t6;
  if (t8) {
   t4 = t3;
   t9 = (*t1);
   t10 = t9.ptr;
   t10 = (uint8_t const *)(((uintptr_t)t10) + (t4*sizeof(uint8_t)));
   t11 = (uint8_t const (*)[8])t10;
   memcpy(t12, (const char *)t11, sizeof(uint8_t[8]));
   memcpy(&t4, &t12, sizeof(uintptr_t));
   t4 = zig_wrap_u64(t4, UINT8_C(64));
   t4 = t4 & (uintptr_t)9259542123273814144ul;
   t6 = t4;
   t8 = t6 != UINT64_C(0);
   if (t8) {
    goto zig_block_2;
   }
   goto zig_block_4;

   zig_block_4:;
   t4 = t2;
   t4 = t4 + (uintptr_t)8ul;
   t2 = t4;
   t4 = t3;
   t4 = t4 + (uintptr_t)8ul;
   t3 = t4;
   goto zig_block_3;
  }
  goto zig_block_2;

  zig_block_3:;
  goto zig_loop_18;

  zig_block_2:;
  t5 = t3;
  t4 = a0.len;
  t7 = t5;
  t6 = t4;
  t8 = t7 < t6;
  if (t8) {
   t4 = t3;
   t13 = a0.ptr[t4];
   t14 = unicode_utf8ByteSequenceLength__6573(t13);
   if (t14.error) {
    t15 = t14.error;
    t16.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
    t16.error = t15;
    return t16;
   }
   t13 = t14.payload;
   t4 = t3;
   t5 = (uintptr_t)t13;
   t5 = t4 + t5;
   t4 = a0.len;
   t6 = t5;
   t7 = t4;
   t8 = t6 > t7;
   if (t8) {
    return (nav__6589_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_TruncatedInput};
   }
   goto zig_block_6;

   zig_block_6:;
   switch (t13) {
    case UINT8_C(1): {
     goto zig_block_7;
    }
    default: {
     t4 = t3;
     t9 = (*t1);
     t10 = t9.ptr;
     t10 = (uint8_t const *)(((uintptr_t)t10) + (t4*sizeof(uint8_t)));
     t4 = (uintptr_t)t13;
     t9.ptr = t10;
     t9.len = t4;
     t17 = unicode_utf8Decode__6579(t9);
     if (t17.error) {
      t15 = t17.error;
      t16.payload = (uintptr_t)0xaaaaaaaaaaaaaaaaul;
      t16.error = t15;
      return t16;
     }
     goto zig_block_7;
    }
   }

   zig_block_7:;
   t4 = t3;
   t5 = (uintptr_t)t13;
   t5 = t4 + t5;
   t3 = t5;
   t5 = t2;
   t5 = t5 + (uintptr_t)1ul;
   t2 = t5;
   goto zig_block_5;
  }
  goto zig_block_5;

  zig_block_5:;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_9;

 zig_block_0:;
 t5 = t2;
 t16.payload = t5;
 t16.error = UINT16_C(0);
 return t16;
}

static uint16_t fmt_formatBuf__anon_4769__6743(nav__6743_39 const a0, struct fmt_FormatOptions__4654 const a1, struct io_Writer__3628 const a2) {
 struct io_Writer__3628 const *t1;
 nav__6743_44 t2;
 uintptr_t t4;
 uintptr_t t5;
 uintptr_t t7;
 nav__6743_48 t6;
 uint64_t t8;
 uint64_t t9;
 struct io_Writer__3628 t10;
 struct io_Writer__3628 t0;
 nav__6743_57 t15;
 nav__6743_39 t17;
 nav__6743_39 t20;
 uint8_t *t19;
 uint32_t t14;
 uint16_t t11;
 uint16_t t12;
 nav__6743_60 t16;
 bool t3;
 uint8_t t18;
 uint8_t t13[4];
 t0 = a2;
 t1 = (struct io_Writer__3628 const *)&t0;
 t2 = a1.width;
 t3 = t2.is_null != true;
 if (t3) {
  t4 = t2.payload;
  t6 = unicode_utf8CountCodepoints__6589(a0);
  t3 = t6.error == UINT16_C(0);
  if (t3) {
   t7 = t6.payload;
   t5 = t7;
   goto zig_block_1;
  }
  t7 = a0.len;
  t5 = t7;
  goto zig_block_1;

  zig_block_1:;
  t8 = t5;
  t9 = t4;
  t3 = t8 < t9;
  if (t3) {
   t5 = t4 - t5;
   t7 = t5;
   goto zig_block_2;
  }
  t7 = (uintptr_t)0ul;
  goto zig_block_2;

  zig_block_2:;
  t9 = t7;
  t3 = t9 == UINT64_C(0);
  if (t3) {
   t10 = (*t1);
   t11 = io_Writer_writeAll__6036(t10, a0);
   memcpy(&t12, &t11, sizeof(uint16_t));
   return t12;
  }
  goto zig_block_3;

  zig_block_3:;
  t14 = a1.fill;
  t15.ptr = &t13[(uintptr_t)0ul];
  t15.len = (uintptr_t)4ul;
  t16 = unicode_utf8Encode__6574(t14, t15);
  t3 = t16.error == UINT16_C(0);
  if (t3) {
   t18 = t16.payload;
   t19 = (uint8_t *)&t13;
   t19 = (uint8_t *)(((uintptr_t)t19) + ((uintptr_t)0ul*sizeof(uint8_t)));
   t5 = (uintptr_t)t18;
   t15.ptr = t19;
   t15.len = t5;
   memcpy(&t20, &t15, sizeof(nav__6743_39));
   t17 = t20;
   goto zig_block_4;

  }
  t12 = t16.error;
  switch (t12) {
   case zig_error_Utf8CannotEncodeSurrogateHalf:
   case zig_error_CodepointTooLarge: {
    t17 = (nav__6743_39){(uint8_t const *)&__anon_4868,(uintptr_t)3ul};
    goto zig_block_4;
   }
   default: zig_unreachable();
  }

  zig_block_4:;
  t18 = a1.alignment;
  switch (t18) {
   case UINT8_C(0): {
    t10 = (*t1);
    t12 = io_Writer_writeAll__6036(t10, a0);
    if (t12) {
     return t12;
    }
    t10 = (*t1);
    t12 = io_Writer_writeBytesNTimes__6040(t10, t17, t7);
    if (t12) {
     return t12;
    }
    goto zig_block_6;
   }
   case UINT8_C(1): {
    t5 = t7 / (uintptr_t)2ul;
    t7 = t7 + (uintptr_t)1ul;
    t7 = t7 / (uintptr_t)2ul;
    t10 = (*t1);
    t12 = io_Writer_writeBytesNTimes__6040(t10, t17, t5);
    if (t12) {
     return t12;
    }
    t10 = (*t1);
    t12 = io_Writer_writeAll__6036(t10, a0);
    if (t12) {
     return t12;
    }
    t10 = (*t1);
    t12 = io_Writer_writeBytesNTimes__6040(t10, t17, t7);
    if (t12) {
     return t12;
    }
    goto zig_block_6;
   }
   case UINT8_C(2): {
    t10 = (*t1);
    t12 = io_Writer_writeBytesNTimes__6040(t10, t17, t7);
    if (t12) {
     return t12;
    }
    t10 = (*t1);
    t12 = io_Writer_writeAll__6036(t10, a0);
    if (t12) {
     return t12;
    }
    goto zig_block_6;
   }
   default: zig_unreachable();
  }

  zig_block_6:;
  goto zig_block_0;
 }
 t10 = (*t1);
 t12 = io_Writer_writeAll__6036(t10, a0);
 if (t12) {
  return t12;
 }
 goto zig_block_0;

 zig_block_0:;
 return 0;
}

static nav__4125_38 io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29_write__4125(struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const a0, nav__4125_42 const a1) {
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *t1;
 nav__4125_42 const *t3;
 uintptr_t t4;
 uintptr_t t10;
 uint64_t t5;
 uint64_t t11;
 uintptr_t *t7;
 nav__4125_50 *t8;
 nav__4125_50 t9;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t12;
 struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *t0;
 uint8_t *t13;
 nav__4125_42 t14;
 nav__4125_42 t2;
 uint8_t const *t15;
 nav__4125_38 t16;
 bool t6;
 t0 = a0;
 t1 = (struct io_fixed_buffer_stream_FixedBufferStream_28_5b_5du8_29__2133 *const *)&t0;
 t2 = a1;
 t3 = (nav__4125_42 const *)&t2;
 t4 = a1.len;
 t5 = t4;
 t6 = t5 == UINT64_C(0);
 if (t6) {
  return (nav__4125_38){(uintptr_t)0ul,0};
 }
 goto zig_block_0;

 zig_block_0:;
 t7 = (uintptr_t *)&a0->pos;
 t4 = (*t7);
 t8 = (nav__4125_50 *)&a0->buffer;
 t9 = (*t8);
 t10 = t9.len;
 t5 = t4;
 t11 = t10;
 t6 = t5 >= t11;
 if (t6) {
  return (nav__4125_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NoSpaceLeft};
 }
 goto zig_block_1;

 zig_block_1:;
 t8 = (nav__4125_50 *)&a0->buffer;
 t9 = (*t8);
 t10 = t9.len;
 t7 = (uintptr_t *)&a0->pos;
 t4 = (*t7);
 t4 = t10 - t4;
 t10 = a1.len;
 t10 = (t4 < t10) ? t4 : t10;
 t11 = t10;
 t12 = (*t1);
 t8 = (nav__4125_50 *)&t12->buffer;
 t7 = (uintptr_t *)&a0->pos;
 t10 = (*t7);
 t9 = (*t8);
 t13 = t9.ptr;
 t13 = (uint8_t *)(((uintptr_t)t13) + (t10*sizeof(uint8_t)));
 t10 = t11;
 t9.ptr = t13;
 t9.len = t10;
 t14 = (*t3);
 t15 = t14.ptr;
 t15 = (uint8_t const *)(((uintptr_t)t15) + ((uintptr_t)0ul*sizeof(uint8_t)));
 t10 = t11;
 t14.ptr = t15;
 t14.len = t10;
 t15 = t14.ptr;
 if (t9.len != 0) memcpy(t9.ptr, t15, t9.len * sizeof(uint8_t));
 t12 = (*t1);
 t7 = (uintptr_t *)&t12->pos;
 t10 = (*t7);
 t4 = t11;
 t4 = t10 + t4;
 (*t7) = t4;
 t6 = t11 == UINT64_C(0);
 if (t6) {
  return (nav__4125_38){(uintptr_t)0xaaaaaaaaaaaaaaaaul,zig_error_NoSpaceLeft};
 }
 goto zig_block_2;

 zig_block_2:;
 t4 = t11;
 t16.payload = t4;
 t16.error = UINT16_C(0);
 return t16;
}

static nav__4035_38 io_dummyWrite__4035(nav__4035_40 const a0) {
 uintptr_t t0;
 nav__4035_38 t1;
 t0 = a0.len;
 t1.payload = t0;
 t1.error = UINT16_C(0);
 return t1;
}

static nav__6579_38 unicode_utf8Decode__6579(nav__6579_40 const a0) {
 nav__6579_40 const *t1;
 uintptr_t t2;
 nav__6579_40 t7;
 nav__6579_40 t0;
 uint8_t const *t8;
 uint8_t const (*t9)[2];
 uint8_t const (*t13)[3];
 uint8_t const (*t16)[4];
 nav__6579_38 t3;
 nav__6579_38 t6;
 nav__6579_38 t12;
 uint32_t t5;
 uint8_t t4;
 uint8_t t10[2];
 nav__6579_48 t11;
 uint8_t t14[3];
 nav__6579_52 t15;
 uint8_t t17[4];
 nav__6579_56 t18;
 t0 = a0;
 t1 = (nav__6579_40 const *)&t0;
 t2 = a0.len;
 switch (t2) {
  case (uintptr_t)1ul: {
   t4 = a0.ptr[(uintptr_t)0ul];
   t5 = (uint32_t)t4;
   t6.payload = t5;
   t6.error = UINT16_C(0);
   t3 = t6;
   goto zig_block_0;
  }
  case (uintptr_t)2ul: {
   t7 = (*t1);
   t8 = t7.ptr;
   t8 = (uint8_t const *)(((uintptr_t)t8) + ((uintptr_t)0ul*sizeof(uint8_t)));
   t9 = (uint8_t const (*)[2])t8;
   memcpy(t10, (const char *)t9, sizeof(uint8_t[2]));
   memcpy(t11.array, t10, sizeof(nav__6579_48));
   t6 = unicode_utf8Decode2__6581(t11);
   memcpy(&t12, &t6, sizeof(nav__6579_38));
   t3 = t12;
   goto zig_block_0;
  }
  case (uintptr_t)3ul: {
   t7 = (*t1);
   t8 = t7.ptr;
   t8 = (uint8_t const *)(((uintptr_t)t8) + ((uintptr_t)0ul*sizeof(uint8_t)));
   t13 = (uint8_t const (*)[3])t8;
   memcpy(t14, (const char *)t13, sizeof(uint8_t[3]));
   memcpy(t15.array, t14, sizeof(nav__6579_52));
   t12 = unicode_utf8Decode3__6583(t15);
   memcpy(&t6, &t12, sizeof(nav__6579_38));
   t3 = t6;
   goto zig_block_0;
  }
  case (uintptr_t)4ul: {
   t7 = (*t1);
   t8 = t7.ptr;
   t8 = (uint8_t const *)(((uintptr_t)t8) + ((uintptr_t)0ul*sizeof(uint8_t)));
   t16 = (uint8_t const (*)[4])t8;
   memcpy(t17, (const char *)t16, sizeof(uint8_t[4]));
   memcpy(t18.array, t17, sizeof(nav__6579_56));
   t12 = unicode_utf8Decode4__6587(t18);
   memcpy(&t6, &t12, sizeof(nav__6579_38));
   t3 = t6;
   goto zig_block_0;
  }
  default: {
   zig_unreachable();
  }
 }

 zig_block_0:;
 return t3;
}

static nav__6572_38 unicode_utf8CodepointSequenceLength__6572(uint32_t const a0) {
 bool t0;
 t0 = a0 < UINT32_C(128);
 if (t0) {
  return (nav__6572_38){0,UINT8_C(1)};
 }
 goto zig_block_0;

 zig_block_0:;
 t0 = a0 < UINT32_C(2048);
 if (t0) {
  return (nav__6572_38){0,UINT8_C(2)};
 }
 goto zig_block_1;

 zig_block_1:;
 t0 = a0 < UINT32_C(65536);
 if (t0) {
  return (nav__6572_38){0,UINT8_C(3)};
 }
 goto zig_block_2;

 zig_block_2:;
 t0 = a0 < UINT32_C(1114112);
 if (t0) {
  return (nav__6572_38){0,UINT8_C(4)};
 }
 goto zig_block_3;

 zig_block_3:;
 return (nav__6572_38){zig_error_CodepointTooLarge,UINT8_C(0x2)};
}

static nav__6744_38 unicode_utf8EncodeImpl__anon_4900__6744(uint32_t const a0, nav__6744_40 const a1) {
 nav__6744_40 const *t1;
 uintptr_t t5;
 uint64_t t6;
 uint64_t t7;
 nav__6744_40 t9;
 nav__6744_40 t0;
 uint8_t *t10;
 uint32_t t12;
 nav__6744_38 t2;
 uint16_t t3;
 uint8_t t4;
 uint8_t t11;
 bool t8;
 t0 = a1;
 t1 = (nav__6744_40 const *)&t0;
 t2 = unicode_utf8CodepointSequenceLength__6572(a0);
 if (t2.error) {
  t3 = t2.error;
  t2.payload = UINT8_C(0x2);
  t2.error = t3;
  return t2;
 }
 t4 = t2.payload;
 t5 = a1.len;
 t6 = t5;
 t7 = (uint64_t)t4;
 t8 = t6 >= t7;
 debug_assert__179(t8);
 switch (t4) {
  case UINT8_C(1): {
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)0ul];
   t11 = (uint8_t)a0;
   (*t10) = t11;
   goto zig_block_0;
  }
  case UINT8_C(2): {
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)0ul];
   t12 = zig_shr_u32(a0, UINT8_C(6));
   t12 = UINT32_C(192) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)1ul];
   t12 = a0 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   goto zig_block_0;
  }
  case UINT8_C(3): {
   t8 = unicode_isSurrogateCodepoint__6652(a0);
   if (t8) {
    return (nav__6744_38){zig_error_Utf8CannotEncodeSurrogateHalf,UINT8_C(0x2)};
   }
   goto zig_block_1;

   zig_block_1:;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)0ul];
   t12 = zig_shr_u32(a0, UINT8_C(12));
   t12 = UINT32_C(224) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)1ul];
   t12 = zig_shr_u32(a0, UINT8_C(6));
   t12 = t12 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)2ul];
   t12 = a0 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   goto zig_block_0;
  }
  case UINT8_C(4): {
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)0ul];
   t12 = zig_shr_u32(a0, UINT8_C(18));
   t12 = UINT32_C(240) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)1ul];
   t12 = zig_shr_u32(a0, UINT8_C(12));
   t12 = t12 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)2ul];
   t12 = zig_shr_u32(a0, UINT8_C(6));
   t12 = t12 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   t9 = (*t1);
   t10 = &t9.ptr[(uintptr_t)3ul];
   t12 = a0 & UINT32_C(63);
   t12 = UINT32_C(128) | t12;
   t11 = (uint8_t)t12;
   (*t10) = t11;
   goto zig_block_0;
  }
  default: {
   zig_unreachable();
  }
 }

 zig_block_0:;
 t2.payload = t4;
 t2.error = UINT16_C(0);
 return t2;
}

static nav__6574_38 unicode_utf8Encode__6574(uint32_t const a0, nav__6574_40 const a1) {
 nav__6574_38 t0;
 nav__6574_38 t1;
 t0 = unicode_utf8EncodeImpl__anon_4900__6744(a0, a1);
 memcpy(&t1, &t0, sizeof(nav__6574_38));
 return t1;
}

static uint16_t io_Writer_writeBytesNTimes__6040(struct io_Writer__3628 const a0, nav__6040_40 const a1, uintptr_t const a2) {
 struct io_Writer__3628 const *t1;
 uintptr_t t3;
 uintptr_t t2;
 uint64_t t4;
 uint64_t t5;
 struct io_Writer__3628 t7;
 struct io_Writer__3628 t0;
 uint16_t t8;
 bool t6;
 t0 = a0;
 t1 = (struct io_Writer__3628 const *)&t0;
 t2 = (uintptr_t)0ul;
 zig_loop_9:
 t3 = t2;
 t4 = t3;
 t5 = a2;
 t6 = t4 < t5;
 if (t6) {
  t7 = (*t1);
  t8 = io_Writer_writeAll__6036(t7, a1);
  if (t8) {
   return t8;
  }
  t3 = t2;
  t3 = t3 + (uintptr_t)1ul;
  t2 = t3;
  goto zig_block_1;
 }
 goto zig_block_0;

 zig_block_1:;
 goto zig_loop_9;

 zig_block_0:;
 return 0;
}

static nav__6581_38 unicode_utf8Decode2__6581(nav__6581_40 const a0) {
 uint32_t t3;
 uint32_t t4;
 uint32_t t2;
 nav__6581_38 t5;
 uint8_t t0;
 bool t1;
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(224);
 t1 = t0 == UINT8_C(192);
 debug_assert__179(t1);
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(31);
 t3 = (uint32_t)t0;
 t2 = t3;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6581_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = t2;
 t3 = zig_shlw_u32(t3, UINT8_C(6), UINT8_C(21));
 t2 = t3;
 t3 = t2;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(63);
 t4 = (uint32_t)t0;
 t4 = t3 | t4;
 t2 = t4;
 t4 = t2;
 t1 = t4 < UINT32_C(128);
 if (t1) {
  return (nav__6581_38){UINT32_C(0xaaaaa),zig_error_Utf8OverlongEncoding};
 }
 goto zig_block_1;

 zig_block_1:;
 t4 = t2;
 t5.payload = t4;
 t5.error = UINT16_C(0);
 return t5;
}

static nav__6583_38 unicode_utf8Decode3__6583(nav__6583_40 const a0) {
 nav__6583_38 t1;
 uint32_t t3;
 uint16_t t2;
 nav__6583_40 t0;
 bool t4;
 bool t5;
 memcpy(t0.array, a0.array, sizeof(nav__6583_40));
 t1 = unicode_utf8Decode3AllowSurrogateHalf__6585(t0);
 if (t1.error) {
  t2 = t1.error;
  t1.payload = UINT32_C(0xaaaaa);
  t1.error = t2;
  return t1;
 }
 t3 = t1.payload;
 t4 = UINT32_C(55296) <= t3;
 if (t4) {
  t4 = t3 <= UINT32_C(57343);
  t5 = t4;
  goto zig_block_1;
 }
 t5 = false;
 goto zig_block_1;

 zig_block_1:;
 if (t5) {
  return (nav__6583_38){UINT32_C(0xaaaaa),zig_error_Utf8EncodesSurrogateHalf};
 }
 goto zig_block_0;

 zig_block_0:;
 t1.payload = t3;
 t1.error = UINT16_C(0);
 return t1;
}

static nav__6587_38 unicode_utf8Decode4__6587(nav__6587_40 const a0) {
 uint32_t t3;
 uint32_t t4;
 uint32_t t2;
 nav__6587_38 t5;
 uint8_t t0;
 bool t1;
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(248);
 t1 = t0 == UINT8_C(240);
 debug_assert__179(t1);
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(7);
 t3 = (uint32_t)t0;
 t2 = t3;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6587_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = t2;
 t3 = zig_shlw_u32(t3, UINT8_C(6), UINT8_C(21));
 t2 = t3;
 t3 = t2;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(63);
 t4 = (uint32_t)t0;
 t4 = t3 | t4;
 t2 = t4;
 t0 = a0.array[(uintptr_t)2ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6587_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_1;

 zig_block_1:;
 t4 = t2;
 t4 = zig_shlw_u32(t4, UINT8_C(6), UINT8_C(21));
 t2 = t4;
 t4 = t2;
 t0 = a0.array[(uintptr_t)2ul];
 t0 = t0 & UINT8_C(63);
 t3 = (uint32_t)t0;
 t3 = t4 | t3;
 t2 = t3;
 t0 = a0.array[(uintptr_t)3ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6587_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_2;

 zig_block_2:;
 t3 = t2;
 t3 = zig_shlw_u32(t3, UINT8_C(6), UINT8_C(21));
 t2 = t3;
 t3 = t2;
 t0 = a0.array[(uintptr_t)3ul];
 t0 = t0 & UINT8_C(63);
 t4 = (uint32_t)t0;
 t4 = t3 | t4;
 t2 = t4;
 t4 = t2;
 t1 = t4 < UINT32_C(65536);
 if (t1) {
  return (nav__6587_38){UINT32_C(0xaaaaa),zig_error_Utf8OverlongEncoding};
 }
 goto zig_block_3;

 zig_block_3:;
 t4 = t2;
 t1 = t4 > UINT32_C(1114111);
 if (t1) {
  return (nav__6587_38){UINT32_C(0xaaaaa),zig_error_Utf8CodepointTooLarge};
 }
 goto zig_block_4;

 zig_block_4:;
 t4 = t2;
 t5.payload = t4;
 t5.error = UINT16_C(0);
 return t5;
}

static bool unicode_isSurrogateCodepoint__6652(uint32_t const a0) {
 bool t0;
 switch (a0) {
  default: if ((a0 >= UINT32_C(55296) && a0 <= UINT32_C(57343))) {
   t0 = true;
   goto zig_block_0;
  }{
   t0 = false;
   goto zig_block_0;
  }
 }

 zig_block_0:;
 return t0;
}

static nav__6585_38 unicode_utf8Decode3AllowSurrogateHalf__6585(nav__6585_40 const a0) {
 uint32_t t3;
 uint32_t t4;
 uint32_t t2;
 nav__6585_38 t5;
 uint8_t t0;
 bool t1;
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(240);
 t1 = t0 == UINT8_C(224);
 debug_assert__179(t1);
 t0 = a0.array[(uintptr_t)0ul];
 t0 = t0 & UINT8_C(15);
 t3 = (uint32_t)t0;
 t2 = t3;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6585_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_0;

 zig_block_0:;
 t3 = t2;
 t3 = zig_shlw_u32(t3, UINT8_C(6), UINT8_C(21));
 t2 = t3;
 t3 = t2;
 t0 = a0.array[(uintptr_t)1ul];
 t0 = t0 & UINT8_C(63);
 t4 = (uint32_t)t0;
 t4 = t3 | t4;
 t2 = t4;
 t0 = a0.array[(uintptr_t)2ul];
 t0 = t0 & UINT8_C(192);
 t1 = t0 != UINT8_C(128);
 if (t1) {
  return (nav__6585_38){UINT32_C(0xaaaaa),zig_error_Utf8ExpectedContinuation};
 }
 goto zig_block_1;

 zig_block_1:;
 t4 = t2;
 t4 = zig_shlw_u32(t4, UINT8_C(6), UINT8_C(21));
 t2 = t4;
 t4 = t2;
 t0 = a0.array[(uintptr_t)2ul];
 t0 = t0 & UINT8_C(63);
 t3 = (uint32_t)t0;
 t3 = t4 | t3;
 t2 = t3;
 t3 = t2;
 t1 = t3 < UINT32_C(2048);
 if (t1) {
  return (nav__6585_38){UINT32_C(0xaaaaa),zig_error_Utf8OverlongEncoding};
 }
 goto zig_block_2;

 zig_block_2:;
 t3 = t2;
 t5.payload = t3;
 t5.error = UINT16_C(0);
 return t5;
}

static uint64_t const builtin_zig_backend__242 = UINT64_C(3);

static bool const start_simplified_logic__109 = false;

static uint8_t const builtin_output_mode__243 = UINT8_C(1);

static uint8_t const builtin_link_mode__244 = UINT8_C(0);

static bool const builtin_link_libc__254 = true;

static bool const posix_use_libc__1196 = true;

static struct Target_Os__563 const builtin_os__250 = {{ .linux = {{{6ul,14ul,6ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}},{6ul,14ul,6ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}}},{2ul,39ul,0ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}},UINT32_C(14)} },UINT8_C(9)};

static uint8_t const c_native_os__1638 = UINT8_C(9);

static struct Target_DynamicLinker__737 const Target_DynamicLinker_none__3360 = {"\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252",UINT8_C(0)};

static bool const builtin_is_test__246 = false;

static uint8_t *heap_CAllocator_alloc__3484(void *const a0, uintptr_t const a1, uint8_t const a2, uintptr_t const a3) {
 uint64_t t0;
 uint8_t *t2;
 bool t1;
 (void)a0;
 (void)a3;
 t0 = a1;
 t1 = t0 > UINT64_C(0);
 debug_assert__179(t1);
 t2 = heap_CAllocator_alignedAlloc__3481(a1, a2);
 return t2;
}

static bool heap_CAllocator_resize__3485(void *const a0, nav__3485_40 const a1, uint8_t const a2, uintptr_t const a3, uintptr_t const a4) {
 uintptr_t t0;
 uint64_t t1;
 uint64_t t2;
 uint8_t *t4;
 bool t3;
 (void)a0;
 (void)a2;
 (void)a4;
 t0 = a1.len;
 t1 = a3;
 t2 = t0;
 t3 = t1 <= t2;
 if (t3) {
  return true;
 }
 goto zig_block_0;

 zig_block_0:;
 t4 = a1.ptr;
 t0 = heap_CAllocator_alignedAllocSize__3483(t4);
 t2 = a3;
 t1 = t0;
 t3 = t2 <= t1;
 if (t3) {
  return true;
 }
 goto zig_block_1;

 zig_block_1:;
 return false;
}

static uint8_t *heap_CAllocator_remap__3486(void *const a0, nav__3486_40 const a1, uint8_t const a2, uintptr_t const a3, uintptr_t const a4) {
 uint8_t *t0;
 uint8_t *t2;
 uint8_t *t3;
 bool t1;
 t1 = heap_CAllocator_resize__3485(a0, a1, a2, a3, a4);
 if (t1) {
  t2 = a1.ptr;
  t3 = (uint8_t *)t2;
  t0 = t3;
  goto zig_block_0;
 }
 t0 = NULL;
 goto zig_block_0;

 zig_block_0:;
 return t0;
}

static void heap_CAllocator_free__3487(void *const a0, nav__3487_40 const a1, uint8_t const a2, uintptr_t const a3) {
 uint8_t *t0;
 (void)a0;
 (void)a2;
 (void)a3;
 t0 = a1.ptr;
 heap_CAllocator_alignedFree__3482(t0);
 return;
}

static uint8_t *heap_CAllocator_alignedAlloc__3481(uintptr_t const a0, uint8_t const a1) {
 uint8_t const *t1;
 uintptr_t t3;
 uint64_t t4;
 void *t9;
 void *t5;
 uint8_t *t10;
 int t6;
 int32_t t7;
 uint8_t t2;
 uint8_t t0;
 bool t8;
 t0 = a1;
 t1 = (uint8_t const *)&t0;
 t2 = (*t1);
 t3 = mem_Alignment_toByteUnits__627(t2);
 t3 = ((uintptr_t)8ul > t3) ? (uintptr_t)8ul : t3;
 t4 = t3;
 t3 = t4;
 t6 = posix_memalign(&t5, t3, a0);
 t7 = t6;
 t8 = t7 != INT32_C(0);
 if (t8) {
  return NULL;
 }
 goto zig_block_0;

 zig_block_0:;
 t9 = t5;
 t10 = (uint8_t *)t9;
 return t10;
}

static uintptr_t heap_CAllocator_alignedAllocSize__3483(uint8_t *const a0) {
 void const *t0;
 void const *t1;
 uintptr_t t2;
 t0 = (void const *)a0;
 t1 = t0;
 t2 = malloc_usable_size(t1);
 return t2;
}

static void heap_CAllocator_alignedFree__3482(uint8_t *const a0) {
 void *t0;
 void *t1;
 t0 = (void *)a0;
 t1 = t0;
 free(t1);
 return;
}

static uintptr_t mem_Alignment_toByteUnits__627(uint8_t const a0) {
 uintptr_t t1;
 uint8_t t0;
 t0 = a0;
 t1 = zig_shlw_u64((uintptr_t)1ul, t0, UINT8_C(64));
 return t1;
}

static struct mem_Allocator__206 const heap_c_allocator__3449 = {((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)&heap_CAllocator_vtable__3476)};

static struct mem_Allocator__206 const c_allocator__230 = {((void *)(uintptr_t)0xaaaaaaaaaaaaaaaaul),((struct mem_Allocator_VTable__209 const *)&heap_CAllocator_vtable__3476)};

static struct Target_Cpu_Feature_Set__754 const Target_Cpu_Feature_Set_empty__3405 = {{0ul,0ul,0ul,0ul,0ul}};

static struct Target_Cpu__722 const builtin_cpu__249 = {((struct Target_Cpu_Model__749 const *)&Target_x86_cpu_alderlake__3508),{{9528139926881771760ul,4854352097251152009ul,15835263048482096792ul,3ul,0ul}},UINT8_C(42)};

static uint8_t const builtin_abi__248 = UINT8_C(1);

static uint8_t const builtin_object_format__252 = UINT8_C(0);

static struct Target__561 const builtin_target__251 = {{((struct Target_Cpu_Model__749 const *)&Target_x86_cpu_alderlake__3508),{{9528139926881771760ul,4854352097251152009ul,15835263048482096792ul,3ul,0ul}},UINT8_C(42)},{{ .linux = {{{6ul,14ul,6ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}},{6ul,14ul,6ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}}},{2ul,39ul,0ul,{NULL,0xaaaaaaaaaaaaaaaaul},{NULL,0xaaaaaaaaaaaaaaaaul}},UINT32_C(14)} },UINT8_C(9)},UINT8_C(1),UINT8_C(0),{"/nix/store/p9kdj55g5l39nbrxpjyz5wc1m0s7rzsx-glibc-2.40-66/lib/ld-linux-x86-64.so.2\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252\252",UINT8_C(82)}};

static struct builtin_CallingConvention__261 const builtin_CallingConvention_c__830 = {{ .x86_64_sysv = {{UINT64_C(0xaaaaaaaaaaaaaaaa),true}} },UINT8_C(4)};

static uint8_t const (*const shared_RUNNER_CTL_FIFO__3612)[21] = &__anon_1800;

static uint8_t const (*const shared_RUNNER_ACK_FIFO__3613)[21] = &__anon_1860;

static uint8_t const mem_native_endian__353 = UINT8_C(1);

static uint8_t const fs_path_native_os__3751 = UINT8_C(9);

static uint8_t const fs_native_os__1008 = UINT8_C(9);

static uint8_t const fs_Dir_native_os__3735 = UINT8_C(9);

static uint8_t const os_linux_native_arch__2544 = UINT8_C(42);

static int const cimport_EINTR__5751 = 4;

static uint8_t const builtin_mode__253 = UINT8_C(3);

static bool const debug_runtime_safety__159 = false;

static bool const fs_Dir_have_flock__3736 = true;

static bool const fs_File_is_windows__1184 = false;

static uint8_t const posix_native_os__1194 = UINT8_C(9);

static bool const posix_lfs64_abi__1624 = true;

static uint8_t const c_native_abi__1636 = UINT8_C(1);

static bool const posix_unexpected_error_tracing__1625 = false;

static bool const mem_backend_supports_vectors__381 = true;

static bool const builtin_valgrind_support__257 = false;

static bool const debug_default_enable_segfault_handler__204 = false;

static uint8_t const log_default_level__6277 = UINT8_C(0);

static struct std_Options__3995 const std_options__97 = {{0xaaaaaaaaaaaaaaaaul,true},{0xaaaaaaaaaaaaaaaaul,true},3ul,false,UINT8_C(0),false,true,false,false,false,UINT8_C(2)};

static nav__3465_38 const heap_page_size_min_default__3465 = {4096ul,false};

static uintptr_t const heap_page_size_min__3444 = 4096ul;

static uint16_t const fmt_max_format_args__6345 = UINT16_C(32);

static uint8_t const (*const fmt_ANY__6348)[4] = &__anon_4742;

static uint8_t const unicode_native_endian__6570 = UINT8_C(1);

static uint32_t const unicode_replacement_character__6571 = UINT32_C(65533);

static bool const os_linux_extern_getauxval__2608 = false;

static struct mem_Allocator_VTable__209 const heap_CAllocator_vtable__3476 = {&heap_CAllocator_alloc__3484,&heap_CAllocator_resize__3485,&heap_CAllocator_remap__3486,&heap_CAllocator_free__3487};

static bool const heap_CAllocator_supports_malloc_size__3477 = true;

static bool const heap_CAllocator_supports_posix_memalign__3479 = true;

static struct Target_Cpu_Model__749 const Target_x86_cpu_alderlake__3508 = {{(uint8_t const *)&__anon_4995,9ul},{(uint8_t const *)&__anon_4995,9ul},{{9516916112185360560ul,4872366487170699401ul,11223533324467504792ul,3ul,0ul}}};
#ifndef WRAPPER_H
#define WRAPPER_H

#include "valgrind.h"
#include "callgrind.h"
#include <stdint.h>

uint8_t running_on_valgrind() {
    return RUNNING_ON_VALGRIND > 0;
}

void callgrind_dump_stats() {
    CALLGRIND_DUMP_STATS;
}

void callgrind_dump_stats_at(uint8_t const * pos_str) {
    CALLGRIND_DUMP_STATS_AT(pos_str);
}

void callgrind_zero_stats() {
    CALLGRIND_ZERO_STATS;
}

void callgrind_start_instrumentation() {
    CALLGRIND_START_INSTRUMENTATION;
}

void callgrind_stop_instrumentation() {
    CALLGRIND_STOP_INSTRUMENTATION;
}

#endif
