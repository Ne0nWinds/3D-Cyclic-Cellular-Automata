import {
  vec3,
  mat4,
} from './wgpu-matrix.module.min.js';

const adapter = await navigator.gpu.requestAdapter();
const limits = adapter.limits;
const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'], requiredLimits: {
	maxBufferSize: limits.maxBufferSize,
	maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
}
});

const canvas = document.getElementById("canvas");
const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();

context.configure({
	device,
	format,
	alphaMode: "opaque"
});

const states = 5;
const threshold = 8;
const cubeSize = 256;
const cubeSizeSq = cubeSize**2;

const workgroupX = 8;
const workgroupY = 4;
const workgroupZ = 2;
const usePackedValues = true;

const packedValueSize = 4;
const packedValueCount = Math.floor(32 / packedValueSize);
const packedValueMask = ((1 << packedValueSize) - 1) | 0;
const packedCubeSize = cubeSize / packedValueCount;

function downloadCanvasPNG(namePrefix = "canvas") {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `${namePrefix}_${stamp}.png`;

  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.download = filename;
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
  }, "image/png");
}

const shaderModule = device.createShaderModule({
	code:
`
struct VSOut {
	@builtin(position) position: vec4f,
	
	@location(0) cube_pos: vec3f
};

struct Constants {
	mvp: mat4x4<f32>,
	time: f32,
	padding0: f32,
	padding1: f32,
	padding2: f32,
};

@group(0) @binding(0) var<uniform> constants : Constants;

@group(1) @binding(0) var<storage, read> readBuffer : array<u32>;
@group(1) @binding(1) var<storage, read_write> writeBuffer : array<u32>;

fn hsl(h: f32, s: f32, l: f32) -> vec3f {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h_ = h * 6.0; // sector 0..6
    let x = c * (1.0 - abs(h_ % 2.0 - 1.0));

    var rgb = vec3f(0.0);

    if (h_ < 1.0) {
        rgb = vec3f(c, x, 0.0);
    } else if (h_ < 2.0) {
        rgb = vec3f(x, c, 0.0);
    } else if (h_ < 3.0) {
        rgb = vec3f(0.0, c, x);
    } else if (h_ < 4.0) {
        rgb = vec3f(0.0, x, c);
    } else if (h_ < 5.0) {
        rgb = vec3f(x, 0.0, c);
    } else {
        rgb = vec3f(c, 0.0, x);
    }

    let m = l - 0.5 * c;
    return rgb + vec3f(m);
}

const cubeSize = ${cubeSize};
const cubeSizeSq = ${cubeSizeSq};

const packedValueSize = ${packedValueSize};
const packedValueCount = ${packedValueCount};
const packedValueMask = ${packedValueMask};
const packedCubeSize = ${packedCubeSize};

@vertex
fn vs_main(@builtin(instance_index) instance_id: u32, @location(0) in_pos: vec3f) -> VSOut {
	var pos = in_pos;

	var out: VSOut;
	out.position = constants.mvp * vec4f(pos, 1.0);
	out.cube_pos = (pos + 0.5) * (1023.0 / 1024.0);
	return out;
}

@fragment
fn fs_main(@location(0) cube_pos : vec3f) -> @location(0) vec4f {

	var current_state = 0u;
	if (usePackedValues) {
		let index3D = vec3f(cube_pos * vec3f(packedCubeSize, cubeSize, cubeSize));
		let flatIndex = u32(index3D.z)*(cubeSize * packedCubeSize) + u32(index3D.y)*packedCubeSize + u32(index3D.x);
		let bitIndex = u32(index3D.x * f32(packedValueCount));
		current_state = (readBuffer[flatIndex] >> (bitIndex * packedValueSize)) & packedValueMask;
	} else {
		let index3D = vec3<i32>(cube_pos * cubeSize);
		let flatIndex = index3D.z*cubeSizeSq + index3D.y*cubeSize + index3D.x;
		current_state = readBuffer[flatIndex];
	}
	let c = hsl(f32(current_state) / f32(states) * 0.5 + 0.5, 0.65, 0.5);
	// let c = vec3f(f32(current_state) / f32(states));
	return vec4f(c, 1.0);
}


const usePackedValues = ${usePackedValues ? "true" : "false"};
const states = ${states};
const threshold = ${threshold};
const range = 1;

const workgroupSize = vec3i(${workgroupX}, ${workgroupY}, ${workgroupZ});

fn wrapCoord(n: i32) -> i32 {
	if (n >= cubeSize) { return n - cubeSize; };
	if (n < 0) { return n + cubeSize; };
	return n;
}

fn wrapCoordPacked(n: i32) -> i32 {
	if (n >= (packedCubeSize)) { return n - (packedCubeSize); };
	if (n < 0) { return n + (packedCubeSize); };
	return n;
}

fn wrapCoords(coords: vec3i) -> vec3i {
	return vec3i(wrapCoord(coords.x), wrapCoord(coords.y), wrapCoord(coords.z));
}

fn cca_basic(id: vec3<u32>, local_id: vec3<u32>, workgroup_id: vec3<u32>) {
	let idx = id.z * cubeSizeSq + id.y * cubeSize + id.x;
	let current_state = readBuffer[idx];
	let next_state = select(current_state + 1, 0, current_state + 1 == states);
	var count = 0u;
	for (var z = -range; z <= range; z += 1) {
		for (var y = -range; y <= range; y += 1) {
			for (var x = -range; x <= range; x += 1) {
				if (z == 0 && y == 0 && x == 0) { continue; }
				let nz = wrapCoord(z + i32(id.z));
				let ny = wrapCoord(y + i32(id.y));
				let nx = wrapCoord(x + i32(id.x));
				let index = (nz*cubeSizeSq) + (ny*cubeSize) + nx;
				if (readBuffer[index] == next_state) {
					count += 1u;
				}
			}
		}
	}
	writeBuffer[idx] = select(current_state, next_state, count >= threshold);
}

fn flatten_index(index: vec3i, dimensions: vec3i) -> i32 {
	return (index.z*(dimensions.x*dimensions.y)) + index.y*dimensions.x + index.x;
}

fn cca_basic_4x(id: vec3<u32>, local_id: vec3<u32>, workgroup_id: vec3<u32>) {
	let idx = id.z * (cubeSize * packedCubeSize) + id.y * packedCubeSize + id.x;
	let packedStates = readBuffer[idx];

	let currentState0 = packedStates & 0xFF;
	let currentState1 = (packedStates >> 8) & 0xFF;
	let currentState2 = (packedStates >> 16) & 0xFF;
	let currentState3 = (packedStates >> 24);

	let nextState0 = select(currentState0 + 1, 0, currentState0 + 1 == states);
	let nextState1 = select(currentState1 + 1, 0, currentState1 + 1 == states);
	let nextState2 = select(currentState2 + 1, 0, currentState2 + 1 == states);
	let nextState3 = select(currentState3 + 1, 0, currentState3 + 1 == states);

	var count0 = 0u;
	var count1 = 0u;
	var count2 = 0u;
	var count3 = 0u;

	for (var z = -range; z <= range; z += 1) {
		let nz = wrapCoord(z + i32(id.z)) * (cubeSize * packedCubeSize);
		for (var y = -range; y <= range; y += 1) {
			let ny = wrapCoord(y + i32(id.y)) * packedCubeSize;

			let packedStates0 = readBuffer[nz + ny + wrapCoordPacked(i32(id.x) - 1)];
			let packedStates1 = readBuffer[nz + ny + wrapCoordPacked(i32(id.x) + 0)];
			let packedStates2 = readBuffer[nz + ny + wrapCoordPacked(i32(id.x) + 1)];

			count0 += u32((packedStates0 >> 24) == nextState0);
			if (z != 0 || y != 0) {
				count0 += u32((packedStates1 & 0xFF) == nextState0);
			}
			count0 += u32(((packedStates1 >> 8) & 0xFF) == nextState0);

			count1 += u32(((packedStates1 >> 0) & 0xFF) == nextState1);
			if (z != 0 || y != 0) {
				count1 += u32(((packedStates1 >> 8) & 0xFF) == nextState1);
			}
			count1 += u32(((packedStates1 >> 16) & 0xFF) == nextState1);

			count2 += u32(((packedStates1 >> 8) & 0xFF) == nextState2);
			if (z != 0 || y != 0) {
				count2 += u32(((packedStates1 >> 16) & 0xFF) == nextState2);
			}
			count2 += u32(((packedStates1 >> 24) & 0xFF) == nextState2);

			count3 += u32(((packedStates1 >> 16) & 0xFF) == nextState3);
			if (z != 0 || y != 0) {
				count3 += u32(((packedStates1 >> 24) & 0xFF) == nextState3);
			}
			count3 += u32(((packedStates2 >> 0) & 0xFF) == nextState3);
		}
	}

	var out = 0u;
	out |= select(currentState0, nextState0, count0 >= threshold);
	out |= select(currentState1, nextState1, count1 >= threshold) << 8;
	out |= select(currentState2, nextState2, count2 >= threshold) << 16;
	out |= select(currentState3, nextState3, count3 >= threshold) << 24;
	writeBuffer[idx] = out;
}

fn cca_packed(id: vec3<u32>, local_id: vec3<u32>, workgroup_id: vec3<u32>) {
	let idx = id.z * (cubeSize * packedCubeSize) + id.y * packedCubeSize + id.x;
	let packedStates = readBuffer[idx];

	var currentStates = array<u32, packedValueCount>();
	var nextStates = array<u32, packedValueCount>();

	for (var i = 0u; i < packedValueCount; i += 1) {
		let state = (packedStates >> (i * packedValueSize)) & packedValueMask;
		let nextState = select(state + 1, 0, state + 1 == states);
		currentStates[i] = state;
		nextStates[i] = nextState;
	}

	let baseX = i32(id.x) - 1;

	var counts = array<u32, packedValueCount>();

	for (var z = -range; z <= range; z += 1) {
		let nz = wrapCoord(z + i32(id.z)) * (cubeSize * packedCubeSize);
		for (var y = -range; y <= range; y += 1) {
			let ny = wrapCoord(y + i32(id.y)) * packedCubeSize;

			var adjacentStates = array<u32, 3>();
			adjacentStates[0] = readBuffer[nz + ny + wrapCoordPacked(baseX)];
			adjacentStates[1] = readBuffer[nz + ny + wrapCoordPacked(baseX + 1)];
			adjacentStates[2] = readBuffer[nz + ny + wrapCoordPacked(baseX + 2)];

			for (var i = 0u; i < packedValueCount; i += 1) {
				var c = 0u;
				let nextState = nextStates[i];

				if (i == 0) {
					let leftState = adjacentStates[0] >> (32 - packedValueSize);
					c += u32(leftState == nextState);
				} else {
					let leftState = (adjacentStates[1] >> ((i - 1) * packedValueSize)) & packedValueMask;
					c += u32(leftState == nextState);
				}

				if (z != 0 || y != 0) {
					c += u32(((adjacentStates[1] >> (i * packedValueSize)) & packedValueMask) == nextState);
				}

				if (i == packedValueCount - 1) {
					let rightState = adjacentStates[2] & packedValueMask;
					c += u32(rightState == nextState);
				} else {
					let rightState = (adjacentStates[1] >> ((i + 1) * packedValueSize)) & packedValueMask;
					c += u32(rightState == nextState);
				}

				counts[i] += c;
			}

		}
	}

	var out = 0u;
	for (var i = 0u; i < packedValueCount; i += 1) {
		let newState = select(currentStates[i], nextStates[i], counts[i] >= threshold);
		out |= newState << (i * packedValueSize);
	}
	writeBuffer[idx] = out;
}

@compute
@workgroup_size(${workgroupX}, ${workgroupY}, ${workgroupZ})
fn cs_main(
	@builtin(global_invocation_id) id: vec3<u32>,
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id : vec3<u32>
) {
	if (usePackedValues) {
		cca_packed(id, local_id, workgroup_id);
	} else {
		cca_basic(id, local_id, workgroup_id);
	}
}
`
});

const vertices = new Float32Array([
  -0.5, -0.5,  0.5,
   0.5, -0.5,  0.5,
   0.5,  0.5,  0.5,
  -0.5, -0.5,  0.5,
   0.5,  0.5,  0.5,
  -0.5,  0.5,  0.5,

   0.5, -0.5, -0.5,
  -0.5, -0.5, -0.5,
  -0.5,  0.5, -0.5,
   0.5, -0.5, -0.5,
  -0.5,  0.5, -0.5,
   0.5,  0.5, -0.5,

   0.5, -0.5,  0.5,
   0.5, -0.5, -0.5,
   0.5,  0.5, -0.5,
   0.5, -0.5,  0.5,
   0.5,  0.5, -0.5,
   0.5,  0.5,  0.5,

  -0.5, -0.5, -0.5,
  -0.5, -0.5,  0.5,
  -0.5,  0.5,  0.5,
  -0.5, -0.5, -0.5,
  -0.5,  0.5,  0.5,
  -0.5,  0.5, -0.5,

  -0.5,  0.5,  0.5,
   0.5,  0.5,  0.5,
   0.5,  0.5, -0.5,
  -0.5,  0.5,  0.5,
   0.5,  0.5, -0.5,
  -0.5,  0.5, -0.5,

  -0.5, -0.5, -0.5,
   0.5, -0.5, -0.5,
   0.5, -0.5,  0.5,
  -0.5, -0.5, -0.5,
   0.5, -0.5,  0.5,
  -0.5, -0.5,  0.5,
]);
const vertexBuffer = device.createBuffer({
	mappedAtCreation: true,
	size: vertices.byteLength,
	usage: GPUBufferUsage.VERTEX
});

{
	const bufferHandle = new Float32Array(vertexBuffer.getMappedRange());
	bufferHandle.set(vertices);
	vertexBuffer.unmap();
}

let bufferIndex = 0;

const uniformBuffer = device.createBuffer({
	size: 32 * 3,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const bufferSize = cubeSize** 2 * (usePackedValues ? packedCubeSize : cubeSize) * 4;
const buffers = [
	device.createBuffer({
		mappedAtCreation: true,
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE,
	}),
	device.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE
	})
];

function mulberry32(seed) {
	let t = seed;
	return function() {
		t |= 0;
		t = t + 0x6D2B79F5 | 0;
		let r = Math.imul(t ^ t >>> 15, 1 | t);
		r = r + Math.imul(r ^ r >>> 7, 61 | r) ^ r;
		return ((r ^ r >>> 14) >>> 0) / 4294967296;
	};
}

const deterministicRandomness = false;
const rand = (deterministicRandomness) ? mulberry32(12345) : Math.random;

{
	const initialBufferData = new Uint32Array(buffers[0].getMappedRange());
	if (usePackedValues) {
		for (let i = 0; i < bufferSize; ++i) {
			let r = 0;
			for (let j = 0; j < packedValueCount; ++j) {
				r |= Math.floor(rand() * states) << (j * packedValueSize);
			}
			initialBufferData[i] = r;
		}
	} else {
		for (let i = 0; i < cubeSize**3; ++i) {
			initialBufferData[i] = rand() * states;
		}
	}
	buffers[0].unmap();
}

let bindGroupIndex = 0;

const group0 = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			buffer: { type: "uniform" }
		}
	]
});

const group1Render = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }
		},
	]
});
const group1Compute = device.createBindGroupLayout({
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }
		},
		{
			binding: 1,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "storage" }
		}
	]
});

const renderPipeline = device.createRenderPipeline({
	layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1Render] }),
	vertex: {
		entryPoint: "vs_main",
		module: shaderModule,
		buffers: [{
			arrayStride: 3 * 4,
			attributes: [
				{ shaderLocation: 0, offset: 0, format: "float32x3" }
			]
		}]
	},
	fragment: {
		entryPoint: "fs_main",
		module: shaderModule,
		targets: [
			{
				format,
				/*
				blend: {
					color: {
						srcFactor: 'src-alpha',
						dstFactor: 'one-minus-src-alpha',
						operation: 'add',
					},
					alpha: {
						srcFactor: 'one',
						dstFactor: 'one-minus-src-alpha',
						operation: 'add',
					},
				},
				writeMask: GPUColorWrite.ALL
				*/
			}
		]
	},
	primitive: {
		topology: "triangle-list",
		frontFace: 'ccw',
		cullMode: 'back'
	},
	depthStencil: {
		format: "depth24plus",
		depthWriteEnabled: false,
		depthCompare: "less"
	}
});

const computePipeline = device.createComputePipeline({
	layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1Compute] }),
	compute: {
		entryPoint: "cs_main",
		module: shaderModule
	}
});

const renderUniformBindGroup = device.createBindGroup({
	layout: group0,
	entries: [
		{ binding: 0, resource: { buffer: uniformBuffer } }
	]
});

const renderBindGroups = [
	device.createBindGroup({
		layout: group1Render,
		entries: [
			{ binding: 0, resource: { buffer: buffers[0] } },
		]
	}),
	device.createBindGroup({
		layout: group1Render,
		entries: [
			{ binding: 0, resource: { buffer: buffers[1] } },
		]
	})
];

const computeBindGroups = [
	device.createBindGroup({
		layout: group1Compute,
		entries: [
			{ binding: 0, resource: { buffer: buffers[0] } },
			{ binding: 1, resource: { buffer: buffers[1] } },
		]
	}),
	device.createBindGroup({
		layout: group1Compute,
		entries: [
			{ binding: 0, resource: { buffer: buffers[1] } },
			{ binding: 1, resource: { buffer: buffers[0] } },
		]
	})
];

const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
const queryBuffer = device.createBuffer({
	size: 256,
	usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
});
const queryBufferCPU = device.createBuffer({
	size: 256,
	usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
});

let depthTexture;
let depthView;

const start = document.timeline.currentTime;

const frameSkip = 1;
let frameIndex = -1;
const useBenchmarks = true;

const samples = 128;
const times = new Float64Array(samples);
let timesIndex = 0;
let sufficientData = false;

let writingBenchmarkResults = false;
async function writeResults(frameIndex) {
	writingBenchmarkResults = true;
	await queryBufferCPU.mapAsync(GPUMapMode.READ);
	const view = new BigInt64Array(queryBufferCPU.getMappedRange());
	const renderPassDuration = view[1] - view[0];
	const computePassDuration = view[3] - view[2];
	queryBufferCPU.unmap();
	writingBenchmarkResults = false;

	const granularity = 1000;
	const renderPassDurationMS = Number(renderPassDuration * BigInt(granularity) / BigInt(1e6)) / granularity;
	const computePassDurationMS = Number(computePassDuration * BigInt(granularity) / BigInt(1e6)) / granularity;
	times[timesIndex] = computePassDurationMS;
	timesIndex = (timesIndex + 1) % samples;
	if (timesIndex == 0) sufficientData = true;

	if (sufficientData) {
		const sorted_times = new Float64Array(samples);
		sorted_times.set(times, 0);
		sorted_times.sort((a, b) => a - b); // ascending

		const median = sorted_times[Math.floor(samples / 2)].toFixed(2);
		const worst = sorted_times[samples - 1].toFixed(2);
		const best = sorted_times[0].toFixed(2);
		const inverse_length = 1.0 / samples;
		const mean = sorted_times.reduce((a, b) => a + b) * inverse_length;
		const variance = sorted_times.reduce((a, b) =>
			a + (b - mean)**2
		) * inverse_length;
		const std_dev = Math.sqrt(variance);
		// console.log(`Compute Pass: ${sorted_times[50].toFixed(2)}ms`);
		console.log(`Compute Pass\nMedian: ${median}\nWorst: ${worst}\nBest:${best}\nMean: ${mean.toFixed(2)}\nStandard Deviation: ${std_dev.toFixed(2)}`);
	}
	// console.log(`Frame ${frameIndex} - Render Pass: ${renderPassDurationMS}ms - Compute Pass: ${computePassDurationMS}ms`);
}

async function frame(currentTime) {

	frameIndex += 1;

	const elapsedTime = (currentTime - start) * (1.0 / 1024.0);

	{
		const eye = vec3.create(Math.sin(0.5) * 1.5, 1.0, Math.cos(0.5) * 1.5);
		const target = vec3.create(0, 0, 0);
		const up = vec3.create(0, 1, 0);
		const view = mat4.lookAt(eye, target, up);

		const fovy = 60 * (Math.PI / 180);
		const proj = mat4.perspective(fovy, canvas.width / canvas.height, 1/8, 1024);

		const model = mat4.identity();
		const pv = mat4.multiply(proj, view);
		const mvp = mat4.multiply(pv, model);

		const upload = new Float32Array(20);
		upload.set(mvp, 0);
		upload[16] = elapsedTime;
		device.queue.writeBuffer(uniformBuffer, 0, upload);
	}

	const view = context.getCurrentTexture().createView();
	const encoder = device.createCommandEncoder();
	{
		const pass = encoder.beginRenderPass({
			colorAttachments: [{
				view,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 0.0, g: 0, b: 0, a: 1 }
			}],
			depthStencilAttachment: {
				view: depthView,
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store"
			},
			timestampWrites: {
				querySet,
				beginningOfPassWriteIndex: 0,
				endOfPassWriteIndex: 1,
			}
		});
		pass.setPipeline(renderPipeline);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.setBindGroup(0, renderUniformBindGroup);
		pass.setBindGroup(1, renderBindGroups[bindGroupIndex]);
		pass.draw(36, 1);
		pass.end();
	}
	if (frameIndex % frameSkip == 0) {
		const pass = encoder.beginComputePass({
			timestampWrites: {
				querySet,
				beginningOfPassWriteIndex: 2,
				endOfPassWriteIndex: 3,
			}
		});
		pass.setPipeline(computePipeline);
		pass.setBindGroup(0, renderUniformBindGroup);
		pass.setBindGroup(1, computeBindGroups[bindGroupIndex]);
		if (usePackedValues) {
			pass.dispatchWorkgroups(cubeSize / workgroupX / packedValueCount, cubeSize / workgroupY, cubeSize / workgroupZ);
		} else {
			pass.dispatchWorkgroups(cubeSize / workgroupX, cubeSize / workgroupY, cubeSize / workgroupZ);
		}
		pass.end();
		bindGroupIndex ^= 1;

		if (useBenchmarks) {
			encoder.resolveQuerySet(querySet, 0, 4, queryBuffer, 0);
			if (!writingBenchmarkResults) {
				encoder.copyBufferToBuffer(queryBuffer, 0, queryBufferCPU, 0, 256);
			}
		}
	}

	device.queue.submit([encoder.finish()]);
	if (useBenchmarks && !writingBenchmarkResults) {
		writeResults(frameIndex);
	}

	requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function resize() {
	const w = window.innerWidth;
	const h = window.innerHeight;
	if (canvas.width != w || canvas.height != h) {
		canvas.width = w;
		canvas.height = h;
		depthTexture = device.createTexture({
			size: [canvas.width, canvas.height],
			format: 'depth24plus',
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});
		depthView = depthTexture.createView();
	}
}
resize();
window.onresize = resize;
