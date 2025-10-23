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

const states = 7;
const threshold = 6;
const cubeSize = 128;
const cubeSizeSq = cubeSize**2;

const workgroupX = 8;
const workgroupY = 4;
const workgroupZ = 2;
const usePackedValues = true;

const packedValueSize = 4;
const packedValueCount = Math.floor(32 / packedValueSize);
const packedValueMask = ((1 << packedValueSize) - 1) | 0;
const packedCubeSize = Math.floor(cubeSize / packedValueCount);

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
	
	@location(0) cube_pos: vec3f,
	@location(1) world_pos: vec3f
};

struct Constants {
	mvp: mat4x4<f32>,
	eye_pos: vec3<f32>,
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

const roseSandPalette = array<vec3<f32>, 10>(
    vec3<f32>(0.298, 0.000, 0.059), // deep wine rose
    vec3<f32>(0.525, 0.063, 0.145), // rich rose red
    vec3<f32>(0.741, 0.251, 0.275), // coral rose
    vec3<f32>(0.878, 0.447, 0.408), // soft clay
    vec3<f32>(0.949, 0.643, 0.510), // warm terracotta
    vec3<f32>(0.925, 0.745, 0.588), // light sand
    vec3<f32>(0.909, 0.800, 0.682), // beige tan
    vec3<f32>(0.949, 0.870, 0.745), // pale cream
    vec3<f32>(0.976, 0.925, 0.847), // warm ivory
    vec3<f32>(0.988, 0.957, 0.909)  // off-white highlight
);

@vertex
fn vs_main(@builtin(instance_index) instance_id: u32, @location(0) in_pos: vec3f) -> VSOut {
	let pos = in_pos;

	var out: VSOut;
	out.position = constants.mvp * vec4f(pos, 1.0);
	out.cube_pos = (pos + 0.5) * (2047.0 / 2048.0);
	out.world_pos = pos;
	return out;
}

@fragment
fn fs_main(@location(0) cube_pos: vec3f, @location(1) world_pos: vec3f) -> @location(0) vec4f {

	var rayPosition = vec3f(cube_pos * vec3f(cubeSize));
	let rayDirection = normalize(world_pos - constants.eye_pos);

	let delta = abs(1.0 / rayDirection);
	let step = sign(rayDirection);

	var sideDistance = vec3f(
		select(fract(rayPosition.x), floor(rayPosition.x + 1.0) - rayPosition.x, rayDirection.x >= 0) * delta.x,
		select(fract(rayPosition.y), floor(rayPosition.y + 1.0) - rayPosition.y, rayDirection.y >= 0) * delta.y,
		select(fract(rayPosition.z), floor(rayPosition.z + 1.0) - rayPosition.z, rayDirection.z >= 0) * delta.z,
	);

	var voxel_state : u32;
	if (usePackedValues) {
		loop {
			let flatIndex = u32(rayPosition.z)*(packedCubeSize*cubeSize) + u32(rayPosition.y)*(packedCubeSize) + (u32(rayPosition.x) / packedValueCount);
			let shiftAmount = (u32(rayPosition.x) % packedValueCount) * packedValueSize;
			voxel_state = (readBuffer[flatIndex] >> shiftAmount) & packedValueMask;

			if (!(voxel_state <= 2)) {
				break;
			}

			if (sideDistance.x < sideDistance.y && sideDistance.x < sideDistance.z) {
				sideDistance.x += delta.x;
				rayPosition.x += step.x;
			} else if (sideDistance.y < sideDistance.z) {
				sideDistance.y += delta.y;
				rayPosition.y += step.y;
			} else {
				sideDistance.z += delta.z;
				rayPosition.z += step.z;
			}
			if (any(rayPosition >= vec3f(cubeSize)) || any(rayPosition < vec3f(0.0))) {
				return vec4f(vec3f(0.0), 1.0);
			}
		}

	} else {
		loop {
			let flatIndex = u32(rayPosition.z)*cubeSizeSq + u32(rayPosition.y)*cubeSize + u32(rayPosition.x);
			voxel_state = readBuffer[flatIndex];

			if (!(voxel_state >= 0 && voxel_state <= 2)) {
				break;
			}

			if (sideDistance.x < sideDistance.y && sideDistance.x < sideDistance.z) {
				sideDistance.x += delta.x;
				rayPosition.x += step.x;
			} else if (sideDistance.y < sideDistance.x && sideDistance.y < sideDistance.z) {
				sideDistance.y += delta.y;
				rayPosition.y += step.y;
			} else {
				sideDistance.z += delta.z;
				rayPosition.z += step.z;
			}
			if (any(rayPosition >= vec3f(cubeSize)) || any(rayPosition < vec3f(0.0))) {
				return vec4f(vec3f(0.0), 1.0);
			}
		}
	}

	// let c = hsl(f32(voxel_state) / f32(states) * 0.5 + 0.5, 0.65, 0.5);
	// let c = vec3f(f32(voxel_state) / f32(states));
	// let c = normalize(world_pos - constants.eye_pos);
	let c = roseSandPalette[voxel_state];
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

	let baseX = i32(id.x) - ((range + packedValueCount - 1) / packedValueCount);

	var counts = array<u32, packedValueCount>();

	for (var z = -range; z <= range; z += 1) {
		let nz = wrapCoord(z + i32(id.z)) * (cubeSize * packedCubeSize);
		for (var y = -range; y <= range; y += 1) {
			let ny = wrapCoord(y + i32(id.y)) * packedCubeSize;

			const len = 1 + ((range + (packedValueCount - 1)) / packedValueCount) * 2;

			var adjacentStates = array<u32, len>();
			for (var i = 0; i < len; i += 1) {
				adjacentStates[i] = readBuffer[nz + ny + wrapCoordPacked(baseX + i)];
			}

			for (var i = 0u; i < packedValueCount; i += 1) {
				var c = 0u;
				let nextState = nextStates[i];

				for (var x = 0u; x <= range * 2; x += 1) {
					const offset = (packedValueCount - (range % packedValueCount));
					let localX = x + i + offset;
					let states = adjacentStates[localX / packedValueCount];
					let shiftAmount = (localX % packedValueCount) * packedValueSize;
					let state = u32(states >> shiftAmount) & packedValueMask;
					c += u32(state == nextState);
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
	size: 256,
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

function degreesToRadians(degrees) {
	return degrees / 180.0 * Math.PI;
}

let runningSimulation = true;

window

window.onkeyup = (e) => {
	if (e.key == " ") {
		runningSimulation = !runningSimulation;
	}
}

async function frame(currentTime) {

	frameIndex += 1;

	const time = (currentTime - start) * (1.0 / 1024.0);

	{
		const eye = vec3.create(Math.sin(degreesToRadians(30 * time)) * 1.5, 1.0, Math.cos(degreesToRadians(30 * time)) * 1.5);
		const target = vec3.create(0, 0, 0);
		const up = vec3.create(0, 1, 0);
		const view = mat4.lookAt(eye, target, up);
		const fovy = 60 * (Math.PI / 180);
		const proj = mat4.perspective(fovy, canvas.width / canvas.height, 1/8, 1024);
		const mvp = mat4.multiply(proj, view);
		const inverse_view = mat4.inverse(view);

		const upload = new Float32Array(16 + 4 + 4);
		upload.set(mvp, 0);
		upload.set(eye, 16);
		upload[20] = time;
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
	if (frameIndex % frameSkip == 0 && runningSimulation) {
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
