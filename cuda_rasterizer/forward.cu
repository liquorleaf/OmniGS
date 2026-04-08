/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx]; // world coordinate
	glm::vec3 dir = pos - campos; // Orientation vector relative to the camera in world coordinates
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		// active_sh_degree to 1 when the coordinates are involved in the color calculation, the higher the level the higher the order of coordinates
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// The resultant color clamped to non-negative. Record this operation and tell the gradient calculation
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	// t is the camera coordinate, there is a limit under perspective, under panoramic there shouldn't be/or only near points cropped out
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// Jacobi from camera coordinates to pixel coordinates
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2DLonlat(const float3& mean, const int width, const int height, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	// [Camera model] t is the camera coordinate, there is a limitation in perspective, in panoramic there should not be/or only crop off the near point.
	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

	// [Camera Model] Camera Coordinate to Pixel Coordinate Jacobi
	float trxztrxz = t.x * t.x + t.z * t.z;
	float trxztrxz_inv = 1.0f / (trxztrxz + 0.0000001f);
	float trxz = sqrtf(trxztrxz);
	float trxz_inv = 1.0f / (trxz + 0.0000001f);
	float trtr = trxztrxz + t.y * t.y;
	float trtr_inv = 1.0f / (trtr + 0.0000001f);

	float W_div_2pi = width * 0.5f * (float)M_1_PI;
	float H_div_pi = height * (float)M_1_PI;

	float dpx_dtx = W_div_2pi * t.z * trxztrxz_inv;
	float dpx_dtz = -W_div_2pi * t.x * trxztrxz_inv;

	float dpy_dtx = -H_div_pi * t.x * t.y * trxz_inv * trtr_inv;
	float dpy_dty = H_div_pi * trxz * trtr_inv;
	float dpy_dtz = -H_div_pi * t.z * t.y * trxz_inv * trtr_inv;

	glm::mat3 J = glm::mat3(
		dpx_dtx, 0.0f, dpx_dtz,
		dpy_dtx, dpy_dty, dpy_dtz,
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
// float inv_cos_lat = trxz_inv * sqrtf(trtr);
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P/*3D Gaus number*/, int D/*active_sh_degree_, range{0,1,2,3}*/, int M/*Sum of dc and rest harmonics (1+15)*/,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs, /*features_dc and features_rest*/
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix, /*Tcw*/
	const float* projmatrix, /*full transform*/
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// preprocessCUDA preprocesses each 3D Gaus
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// Count the radius and the number of covered tiles of Gaus, if it is still zero after preprocessing, then it is not involved in the rendering process
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// Elimination of invisible points; in fact, only points with z<=0.2 in camera coordinates, i.e., points in front of the camera, are eliminated, not involving the camera model
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // screen-space coordiante

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// Generally there is no precompute cov3D, the input is nullptr, so scale and rot are used to compute the 3D covariance
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// Compute the covariance of the projection to 2D screen-space
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// Finding the inverse matrix of conic:2D covariance
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// Calculate the eigenvalues of the 2D covariance -> Calculate the radius of Gaus -> Calculate the range box of Gaus
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// Coloring: SH to RGB
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // Camera coordinates Depth
	radii[idx] = my_radius; // Image Space Radius
	points_xy_image[idx] = point_image; // Image Space Pixel coordinates
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; // packing covariance and opacity
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // Roughly estimated number of covered tiles using boxes
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// Each thread handles one pixel, and pixel threads within the same tile are in the same block (thread block, GPU's feature)
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block(); // The current block is the current tile
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // Total number of blocks horizontally (total number of tiles, = tile_grid.x)
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // Coordinates of the pixel closest to the origin of the current tile
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };        // Coordinates of the pixel furthest from the origin of the current tile
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // Current pixel coordinate = nearest origin pixel + block.thread_index (2D)
	uint32_t pix_id = W * pix.y + pix.x; // Total serial number of the current pixel (the serial number of the pixel when the image is spread to one dimension, first in line and then in column)
	float2 pixf = { (float)pix.x, (float)pix.y };  // Current pixel coordinates

	// Pixels outside the image are invalid, directly done=true, not involved in resampling, not accumulating alpha, but involved in collaborating to get Gaus data from the global to the block
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // The range of instance numbers for the current tile that were previously precomputed.
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // Divide the instances of the current tile into BLOCK_SIZE (= maximum number of pixels per tile) groups for processing
	int toDo = range.y - range.x; // The number of instances of the current pixel that have not yet been used for rendering, initially equal to the total number of instances of the current tile used for rendering

	// Allocate storage for batches of collectively fetched data.
	// Used to store the obtained Gaus data for all pixels within a tile. Array length = maxThreadsPerBlock
	// Shared memory is allocated in thread blocks, so all threads in a block can access the same shared memory https://developer.nvidia.com/zh-cn/blog/using-shared-memory-cuda-cc/
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; // Paper (2) Transmission Ratio
	uint32_t contributor = 0; // Counter: number of Gaus instances involved in composing the current pixel (also counts when actually continue skips an instance but doesn't DONE)
	uint32_t last_contributor = 0; // Auxiliary record, the last instance actually involved in composing the current pixel (not skipped) is the first of all instances for that pixel
	float C[CHANNELS] = { 0 }; // Auxiliary record, cumulative color of current pixel

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count counts the total number of threads in the block that are true when all pixels are done, then all threads in the block break together.
		// Before all pixels are done, the pixels that are already done still participate in the shared data fetch, but they don't add a new contributor and their color doesn't change.
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// In one round, i.e., for the same i, the whole block takes out up to BLOCK_SIZE instances in parallel, 1 for each thread in the block, synchronization: wait until all these instances are taken.
		int progress = i * BLOCK_SIZE + block.thread_rank(); // the rank of instance in current block
		if (range.x + progress < range.y) // range.x + progress = the rank in all instances, only the ones in the current tile range will be taken out
		{
			int coll_id = point_list[range.x + progress]; // The id of the Gaus corresponding to instance
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // The pixel coordinates of the instance's corresponding Gaus
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // instance corresponds to the covariance of Gaus, transparency
		}
		block.sync();

		// Iterate over current batch
		// In one round, i.e., for the same i, each pixel accumulates its own rendering result color using at most BLOCK_SIZE instances just taken out respectively
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j]; // {x=conic[0][0], y=conic[0][1], z=conic[1][1], w=opacity}
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // Exponential term, defined with reference to EWA splatting (20)
			if (power > 0.0f) // Numerical robustness: the exponent of a Gaus distribution function should not be greater than 0
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// Here alpha is not defined in terms of the variables in equation (2), but is directly defined as the opacity property after sigmoid activation * Gaus distribution sample
			float alpha = min(0.99f, con_o.w * exp(power)); // Numerical Robustness 1: preventing division by 0, large alpha clamp to 0.99
			if (alpha < 1.0f / 255.0f) // Numerical robustness 2: preventing division by 0, skipping small alpha
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) // Numerical robustness 3: alpha accumulation stops at 0.9999 instead of 1, i.e., when new T<1-0.9999
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T; // Cumulative multiplication T

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // The background color is also equivalent to a mixing term accrued to equation (3)
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/BLOCK_X * BLOCK_Y)
renderDepthCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	// const float* __restrict__ orig_points,
	// const float* __restrict__ viewmatrix,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); 
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j]; // {x=conic[0][0], y=conic[0][1], z=conic[1][1], w=opacity}
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// int idx = collected_id[j] * 3;
			// float3 p_orig = { orig_points[idx], orig_points[idx + 1], orig_points[idx + 2] };
			// float3 p_view = transformPoint4x3(p_orig, viewmatrix);
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += depths[collected_id[j]] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessLonlatCUDA(int P/*3D Gaus number*/, int D/*active_sh_degree_, range{0,1,2,3}*/, int M/*Sum of dc and rest harmonics (1+15)*/,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs, /*features_dc and features_rest*/
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix, /*Tcw*/
	const glm::vec3* cam_pos,
	const int W, int H,
	int* radii,
// int* radii_x,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// preprocessCUDA preprocesses each 3D Gaus
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// Count the radius and the number of covered tiles of Gaus, if it is still zero after preprocessing, then it is not involved in the rendering process
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// Rejects invisible points; in fact, only points with r<=0.2 in camera coordinates are rejected, not involving proj
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_view;
	if (too_close(p_orig, viewmatrix, p_view))
		return;

	// Transform point by projecting
	// [Camera Model] Calculate the coordinates of the projected model points into the screen-space.
	float2 p_proj = point3ToLonlatScreen(p_view); // screen-space coordinate

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// Generally there is no precompute cov3D, the input is nullptr, so scale and rot are used to compute the 3D covariance
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// [Camera Model] Calculate the covariance of projection to 2D screen-space
	float3 cov = computeCov2DLonlat(p_orig, W, H, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// Finding the inverse matrix of conic:2D covariance
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// Calculate the eigenvalues of the 2D covariance -> Calculate the radius of Gaus -> Calculate the range box of Gaus
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
// float my_radius_y = ceil(3.f * sqrt(min(lambda1, lambda2)));
// float my_radius_x = my_radius * cov.w;
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
// int2 rect_min, rect_max;
	uint2 rect_min, rect_max;
// getRectCyclic(point_image, my_radius_x, my_radius, rect_min, rect_max, grid);
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// Coloring: SH to RGB
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.w; // Camera coordinates Depth
	radii[idx] = my_radius; // Image Space Radius
	points_xy_image[idx] = point_image; // Image space Pixel coordinates
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; // packing covariance and opacity
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // Roughly estimated number of covered tiles using boxes
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::renderDepth(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	// const float* orig_points,
	// const float* viewmatrix,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderDepthCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		// orig_points,
		// viewmatrix,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

// void FORWARD::renderLonlat(
// 	const dim3 grid, dim3 block,
// 	const uint2* ranges,
// 	const uint32_t* point_list,
// 	int W, int H,
// 	const float2* means2D,
// 	const float* colors,
// 	const float4* conic_opacity,
// 	float* final_T,
// 	uint32_t* n_contrib,
// 	const float* bg_color,
// 	float* out_color)
// {
// 	renderLonlatCUDA<NUM_CHANNELS> << <grid, block >> > (
// 		ranges,
// 		point_list,
// 		W, H,
// 		means2D,
// 		colors,
// 		conic_opacity,
// 		final_T,
// 		n_contrib,
// 		bg_color,
// 		out_color);
// }

void FORWARD::preprocessLonlat(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	int* radii,
// int* radii_x,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessLonlatCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		cam_pos,
		W, H,
		radii,
// radii_x,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}