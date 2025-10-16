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
const threshold = 4;
const cubeSize = 256;
const cubeSizeSq = cubeSize**2;

const workgroupX = 8;
const workgroupY = 4;
const workgroupZ = 4;

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
	let index3D = vec3<i32>(cube_pos * cubeSize);
	let flatIndex = index3D.z*cubeSizeSq + index3D.y*cubeSize + index3D.x;

	let current_state = readBuffer[flatIndex];
	// let c = hsl(f32(current_state) / f32(states) * 0.25 + 0.5, 0.75, 0.5);
	let c = vec3f(f32(current_state) / f32(states));
	return vec4f(c, 1.0);
}

fn wrapCoord(n: i32) -> i32 {
	if (n >= cubeSize) { return n - cubeSize; };
	if (n < 0) { return n + cubeSize; };
	return n;
}

const states = ${states};
const threshold = ${threshold};

const workgroupSize = vec3i(${workgroupX}, ${workgroupY}, ${workgroupZ});
const sharedMemorySize = vec3i(${workgroupX + 2}, ${workgroupY + 2}, ${workgroupZ + 2});
const totalSharedMemorySize = sharedMemorySize.x * sharedMemorySize.y * sharedMemorySize.z;

var<workgroup> shared_memory : array<u32, totalSharedMemorySize>;

fn wrapCoords(coords: vec3i) -> vec3i {
	return vec3i(wrapCoord(coords.x), wrapCoord(coords.y), wrapCoord(coords.z));
}

const range = 1;

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
fn cca_shared(id: vec3<u32>, local_id: vec3<u32>, workgroup_id: vec3<u32>) {
	let idx = id.z * cubeSizeSq + id.y * cubeSize + id.x;


	let base = wrapCoords(vec3i(workgroup_id) * workgroupSize - vec3i(1));

	let flat_local_id = (i32(local_id.z)*workgroupSize.x*workgroupSize.y) + (i32(local_id.y) * workgroupSize.x) + i32(local_id.x);
	for (var i = 0; i < totalSharedMemorySize; i += (workgroupSize.x*workgroupSize.y*workgroupSize.z)) {
		let thread_index = i + flat_local_id;
		let z = (thread_index / (sharedMemorySize.x * sharedMemorySize.y));
		let xy = (thread_index % (sharedMemorySize.x * sharedMemorySize.y));
		let y = xy / sharedMemorySize.x;
		let x = xy % sharedMemorySize.x;

		let global_index = wrapCoords(base + vec3i(x, y, z));
		if (thread_index < totalSharedMemorySize) {
			shared_memory[thread_index] = readBuffer[global_index.z * cubeSizeSq + global_index.y * cubeSize + global_index.x];
		}
	}

	workgroupBarrier();

	let shared_idx = vec3i(local_id) + vec3i(range);
	let current_state = shared_memory[(shared_idx.z*sharedMemorySize.x*sharedMemorySize.y) + shared_idx.y*sharedMemorySize.x + shared_idx.x];
	let next_state = select(current_state + 1, 0, current_state + 1 == states);

	var count = 0u;

	for (var z = -range; z <= range; z += 1) {
		for (var y = -range; y <= range; y += 1) {
			for (var x = -range; x <= range; x += 1) {
				if (z == 0 && y == 0 && x == 0) { continue; }

				let nz = z + shared_idx.z;
				let ny = y + shared_idx.y;
				let nx = x + shared_idx.x;
				let index = (nz * sharedMemorySize.x*sharedMemorySize.y) + (ny * sharedMemorySize.x) + nx;
				if (shared_memory[index] == next_state) {
					count += 1;
				}
			}
		}
	}

	writeBuffer[idx] = select(current_state, next_state, count >= threshold);
}

@compute
@workgroup_size(${workgroupX}, ${workgroupY}, ${workgroupZ})
fn cs_main(
	@builtin(global_invocation_id) id: vec3<u32>,
	@builtin(local_invocation_id) local_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id : vec3<u32>
) {
	cca_shared(id, local_id, workgroup_id);
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
const buffers = [
	device.createBuffer({
		mappedAtCreation: true,
		size: cubeSize**3 * 4,
		usage: GPUBufferUsage.STORAGE,
	}),
	device.createBuffer({
		size: cubeSize**3 * 4,
		usage: GPUBufferUsage.STORAGE
	})
];

{
	const initialBufferData = new Uint32Array(buffers[0].getMappedRange());
	for (let i = 0; i < cubeSize**3; ++i) {
		initialBufferData[i] = Math.random() * states;
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
		depthWriteEnabled: true,
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

const frameSkip = 4;
let frameIndex = -1;

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
	console.log(`Frame ${frameIndex} - Render Pass: ${renderPassDurationMS}ms - Compute Pass: ${computePassDurationMS}ms`);
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
	if (frameIndex % frameSkip != 0) {
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
		pass.dispatchWorkgroups(cubeSize / workgroupX, cubeSize / workgroupY, cubeSize / workgroupZ);
		pass.end();
		bindGroupIndex ^= 1;
	}
	encoder.resolveQuerySet(querySet, 0, 4, queryBuffer, 0);
	if (!writingBenchmarkResults) {
		encoder.copyBufferToBuffer(queryBuffer, 0, queryBufferCPU, 0, 256);
	}

	device.queue.submit([encoder.finish()]);
	if (!writingBenchmarkResults) {
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
