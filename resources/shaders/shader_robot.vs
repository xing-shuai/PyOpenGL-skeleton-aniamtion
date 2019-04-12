#version 330 core

const int MAX_JOINTS = 50;
const int MAX_WEIGHTS = 3;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 jointIndex;
layout (location = 3) in vec3 jointWeight;
layout (location = 4) in vec2 aTexCoord;

out vec2 TexCoord;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform mat4 jointTransforms[MAX_JOINTS];

void main()
{

    vec4 totalLocalPos = vec4(0.0);
	//vec4 totalNormal = vec4(0.0);

	for(int i = 0;i < MAX_WEIGHTS; i++){

		mat4 jointTransform = jointTransforms[int(jointIndex[i])];
		vec4 posePosition = jointTransform * vec4(aPos, 1.0);
		totalLocalPos += posePosition * jointWeight[i];

		//vec4 worldNormal = jointTransform * vec4(aNormal, 0.0);
		//totalNormal += worldNormal * jointWeight[i];
	}

	gl_Position = projection * view * model * totalLocalPos;

	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}