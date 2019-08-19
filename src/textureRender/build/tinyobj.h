//
// Copyright 2012-2015, Syoyo Fujita.
//
// Licensed under 2-clause BSD license.
//

//
// version 0.9.20: Fixes creating per-face material using `usemtl`(#68)
// version 0.9.17: Support n-polygon and crease tag(OpenSubdiv extension)
// version 0.9.16: Make tinyobjloader header-only
// version 0.9.15: Change API to handle no mtl file case correctly(#58)
// version 0.9.14: Support specular highlight, bump, displacement and alpha
// map(#53)
// version 0.9.13: Report "Material file not found message" in `err`(#46)
// version 0.9.12: Fix groups being ignored if they have 'usemtl' just before
// 'g' (#44)
// version 0.9.11: Invert `Tr` parameter(#43)
// version 0.9.10: Fix seg fault on windows.
// version 0.9.9 : Replace atof() with custom parser.
// version 0.9.8 : Fix multi-materials(per-face material ID).
// version 0.9.7 : Support multi-materials(per-face material ID) per
// object/group.
// version 0.9.6 : Support Ni(index of refraction) mtl parameter.
//                 Parse transmittance material parameter correctly.
// version 0.9.5 : Parse multiple group name.
//                 Add support of specifying the base path to load material
//                 file.
// version 0.9.4 : Initial support of group tag(g)
// version 0.9.3 : Fix parsing triple 'x/y/z'
// version 0.9.2 : Add more .mtl load support
// version 0.9.1 : Add initial .mtl load support
// version 0.9.0 : Initial
//

//
// Use this in *one* .cc
//   #define TINYOBJLOADER_IMPLEMENTATION
//   #include "tiny_obj_loader.h"
//

#ifndef TINY_OBJ_LOADER_H
#define TINY_OBJ_LOADER_H

#include <string>
#include <vector>
#include <map>

namespace tinyobj {

	typedef struct {
		std::string name;

		float ambient[3];
		float diffuse[3];
		float specular[3];
		float transmittance[3];
		float emission[3];
		float shininess;
		float ior;      // index of refraction
		float dissolve; // 1 == opaque; 0 == fully transparent
						// illumination model (see http://www.fileformat.info/format/material/)
		int illum;

		int dummy; // Suppress padding warning.

		std::string ambient_texname;            // map_Ka
		std::string diffuse_texname;            // map_Kd
		std::string specular_texname;           // map_Ks
		std::string specular_highlight_texname; // map_Ns
		std::string bump_texname;               // map_bump, bump
		std::string displacement_texname;       // disp
		std::string alpha_texname;              // map_d
		std::map<std::string, std::string> unknown_parameter;
	} material_t;

	typedef struct {
		std::string name;

		std::vector<int> intValues;
		std::vector<float> floatValues;
		std::vector<std::string> stringValues;
	} tag_t;

	typedef struct {
		std::vector<float> positions;
		std::vector<float> normals;
		std::vector<float> texcoords;
		std::vector<unsigned int> indices;
		std::vector<unsigned char>
			num_vertices;              // The number of vertices per face. Up to 255.
		std::vector<int> material_ids; // per-face material ID
		std::vector<tag_t> tags;       // SubD tag
	} mesh_t;

	typedef struct {
		std::string name;
		mesh_t mesh;
	} shape_t;

	class MaterialReader {
	public:
		MaterialReader() {}
		virtual ~MaterialReader();

		virtual bool operator()(const std::string &matId,
			std::vector<material_t> &materials,
			std::map<std::string, int> &matMap,
			std::string &err) = 0;
	};

	class MaterialFileReader : public MaterialReader {
	public:
		MaterialFileReader(const std::string &mtl_basepath)
			: m_mtlBasePath(mtl_basepath) {}
		virtual ~MaterialFileReader() {}
		virtual bool operator()(const std::string &matId,
			std::vector<material_t> &materials,
			std::map<std::string, int> &matMap, std::string &err);

	private:
		std::string m_mtlBasePath;
	};

	/// Loads .obj from a file.
	/// 'shapes' will be filled with parsed shape data
	/// The function returns error string.
	/// Returns true when loading .obj become success.
	/// Returns warning and error message into `err`
	/// 'mtl_basepath' is optional, and used for base path for .mtl file.
	/// 'triangulate' is optional, and used whether triangulate polygon face in .obj
	/// or not.
	bool LoadObj(std::vector<shape_t> &shapes,       // [output]
		std::vector<material_t> &materials, // [output]
		std::string &err,                   // [output]
		const char *filename, const char *mtl_basepath = NULL,
		bool triangulate = true);

	/// Loads object from a std::istream, uses GetMtlIStreamFn to retrieve
	/// std::istream for materials.
	/// Returns true when loading .obj become success.
	/// Returns warning and error message into `err`
	bool LoadObj(std::vector<shape_t> &shapes,       // [output]
		std::vector<material_t> &materials, // [output]
		std::string &err,                   // [output]
		std::istream &inStream, MaterialReader &readMatFn,
		bool triangulate = true);

	/// Loads materials into std::map
	void LoadMtl(std::map<std::string, int> &material_map, // [output]
		std::vector<material_t> &materials,       // [output]
		std::istream &inStream);
}




#endif // TINY_OBJ_LOADER_H
