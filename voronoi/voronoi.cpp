/*
Guodong Rong

voronoi.cpp
The main program using JFA to compute Voronoi diagram

Copyright (c) 2005-2006 
School of Computing
National University of Singapore
All Rights Reserved.
*/

#include "voronoi.h"
#include <assert.h>

extern void Energyf(cudaGraphicsResource_t grSite, real* g, real* f, int w, int h, int nsite, const cudaStream_t& stream);
extern void ConvertSites(real* x, cudaGraphicsResource_t gr, int nsite, int screenw, const cudaStream_t& stream);
extern void InitSites(real* x, float* init_sites, int stride, int* nbd, real* l, real* u, int nsite, int screenw);

float* site_list_x = NULL;
float* site_list_x_bar = NULL;
float site_perturb_step = 0;

inline void CopySite(SiteType* dst, float* src, int n) {
	for(int i = 0; i < n; i++) {
		dst[i].vertices[0].x = src[2 * i];
		dst[i].vertices[0].y = src[2 * i + 1];
	}
}

inline void CopySite(float* dst, float* src, int n) {
	memcpy(dst, src, n * 2 * sizeof(float));
}

inline void CopySite(float* dst, SiteType* src, int n) {
	for(int i = 0; i < n; i++) {
		dst[2 * i] = src[i].vertices[0].x;
		dst[2 * i + 1] = src[i].vertices[0].y;
	}
}

/******************************************************************************/
void CheckFramebufferStatus()
{
	GLenum status;
	status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	switch(status) {
		case GL_FRAMEBUFFER_COMPLETE_EXT:
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
			printf("Unsupported framebuffer format\n");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
			printf("Framebuffer incomplete, missing attachment\n");
			break;
		//case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
		//	printf("Framebuffer incomplete, duplicate attachment\n");
		//	break;
		case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
			printf("Framebuffer incomplete, attached images must have same dimensions\n");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
			printf("Framebuffer incomplete, attached images must have same format\n");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
			printf("Framebuffer incomplete, missing draw buffer\n");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
			printf("Framebuffer incomplete, missing read buffer\n");
			break;
		default:
			exit(0);
	}
}

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

void DrawSites(bool FinalDrawSite, real* x, const cudaStream_t& stream)
{
	int i, j;
	int shift;

	glPointSize(1);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId);
/*
	GLvoid* pointer = glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);

	//memcpy(pointer, site_list, sizeof(SiteType) * point_num);
	VertexSiteType* sitelist = (VertexSiteType*)pointer;
	for (i=0; i<site_num; i++)
	{
		sitelist[i].x = (x[i * 2] + 1.0) * 0.5 * (screenwidth-1) + 1;
		sitelist[i].y = (x[i * 2 + 1] + 1.0) * 0.5 * (screenheight-1) + 1;
	}

	glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);*/

	ConvertSites(x, grVbo, point_num * 2, screenwidth, stream);
	

	glVertexPointer(2, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER_ARB, colorboId);
	glColorPointer(4, GL_FLOAT, 0, 0);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, point_num);

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
/*


	glBegin(GL_POINTS);
	for (i=0; i<site_num; i++)
	{
		glColor4fv(site_list[i].color);
		glVertex2f(site_list[i].vertices[0].x, site_list[i].vertices[0].y);
	}
	glEnd();*/
}

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream)
{
	int i,j;
	get_timestamp(start_time_func);
/*
	for (i=0; i<site_num; i++)
	{
		site_list[i].vertices[0].x = (x[i * 2] + 1.0) * 0.5 * (screenwidth-1) + 1;
		site_list[i].vertices[0].y = (x[i * 2 + 1] + 1.0) * 0.5 * (screenheight-1) + 1;
	}*/






	//////////////////////////////////////////////
	// First pass - Render the initial sites    //
	//////////////////////////////////////////////
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, FB_objects);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
		GL_TEXTURE_RECTANGLE_NV, Processed_Texture[0], 0);
	CheckFramebufferStatus();

	for (i=0; i<1; i++)
	{
		glDrawBuffer(fbo_attachments[i]);
		glClearColor(-1, -1, -1, -1);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glDrawBuffer(fbo_attachments[0]);

	cgGLEnableProfile(VertexProfile);
	cgGLEnableProfile(FragmentProfile);

	cgGLBindProgram(VP_DrawSites);
	cgGLBindProgram(FP_DrawSites);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(1, screenwidth+1, 1, screenheight+1);
	glViewport(1, 1, screenwidth, screenheight);

	DrawSites(false, x, stream);

	glReadBuffer(fbo_attachments[0]);
	//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

	//if (numIter==109)
	//{
	//	GLubyte *buffer_screen = new GLubyte[screenwidth*screenheight*4];
	//	char rawname[40];
	//	glReadPixels(1,1,screenwidth,screenheight,GL_RGBA,GL_UNSIGNED_BYTE,buffer_screen);
	//	strcpy(rawname, "a.raw");
	//	std::ofstream myrawfile(rawname, std::ios::binary);
	//	for (j=screenheight-1; j>=0; j--)
	//		for (i=0; i<screenwidth; i++)
	//		{
	//			GLubyte r, g, b, value;
	//			r = (unsigned char) buffer_screen[(j*screenwidth+i)*4];
	//			g = (unsigned char) buffer_screen[(j*screenwidth+i)*4+1];
	//			b = (unsigned char) buffer_screen[(j*screenwidth+i)*4+2];
	//			if (r>0 || g>0 || b>0)
	//				value = 0;
	//			else
	//				value = 255;
	//			myrawfile << value << value << value;
	//		}
	//	myrawfile.close();
	//	delete buffer_screen;
	//}

	Current_Buffer = 1;

#ifdef DEBUG_TIME
	glFinish();
	get_timestamp(end_time);
	elapsed_time = (end_time-start_time);
	total_time += elapsed_time;
	printf("Step 1 time: %f\n", elapsed_time);
#endif

#ifdef DEBUG_TIME
	get_timestamp(start_time);
#endif
	/////////////////////////////////////
	// Second pass - Flood the sites   //
	/////////////////////////////////////
	cgGLBindProgram(VP_Flood);
	cgGLBindProgram(FP_Flood);

	if (VP_Flood_Size != NULL)
		cgSetParameter2f(VP_Flood_Size, screenwidth, screenheight);

	bool ExitLoop = false;
	bool SecondExit;
	int steplength;;
	SecondExit = (additional_passes==0);
	bool PassesBeforeJFA;
	PassesBeforeJFA = (additional_passes_before>0);
	if (PassesBeforeJFA)
		steplength = pow(2.0, (additional_passes_before-1));
	else
		steplength = (screenwidth>screenheight ? screenwidth : screenheight)/2;

	while (!ExitLoop)
	{
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[Current_Buffer], 
			GL_TEXTURE_RECTANGLE_NV, Processed_Texture[Current_Buffer], 0);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[Current_Buffer]);

		glClearColor(-1, -1, -1, -1);
		glClear(GL_COLOR_BUFFER_BIT);

		//Bind & enable shadow map texture
		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);
		if (VP_Flood_Steplength != NULL)
			cgSetParameter1d(VP_Flood_Steplength, steplength);

		glBegin(GL_QUADS);
			glVertex2f(1.0, 1.0);
			glVertex2f(1.0, float(screenheight+1));
			glVertex2f(float(screenwidth+1), float(screenheight+1));
			glVertex2f(float(screenwidth+1), 1.0);
		glEnd();
		glReadBuffer(fbo_attachments[Current_Buffer]);
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

		if (steplength==1 && PassesBeforeJFA)
		{
			steplength = (screenwidth>screenheight ? screenwidth : screenheight)/2;
			PassesBeforeJFA = false;
		}
		else if (steplength>1)
			steplength /= 2;
		else if (SecondExit)
			ExitLoop = true;
		else
		{
			steplength = pow(2.0, (additional_passes-1));
			SecondExit = true;
		}
		Current_Buffer = 1-Current_Buffer;
	}
	//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

#ifdef DEBUG_TIME
	glFinish();
	get_timestamp(end_time);
	elapsed_time = (end_time-start_time);
	total_time += elapsed_time;
	printf("Step 2 time: %f\n", elapsed_time);
#endif
#ifdef DEBUG_TIME
	get_timestamp(start_time);
#endif
	////////////////////////////////
	// Third pass, Compute energy //
	////////////////////////////////
	cgGLBindProgram(VP_Scatter);
	cgGLBindProgram(FP_Scatter);

	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[0], 
		GL_TEXTURE_RECTANGLE_NV, Site_Texture, 0);
	CheckFramebufferStatus();
	glDrawBuffer(fbo_attachments[0]);

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	if (VP_Scatter_Size != NULL)
		cgSetParameter2f(VP_Scatter_Size, screenwidth, screenheight);

	//Bind & enable shadow map texture
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glCallList(ScreenPointsList);
	glDisable(GL_BLEND);
/*

	cutilSafeCall(cudaGraphicsMapResources(1, &grSite, 0));
	cudaArray *in_array; 
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, grSite, 0, 0));
	
	cutilSafeCall(cudaMemcpy2DFromArray(pReadBackValues, screenwidth * sizeof(float) * 4, in_array, sizeof(float) * 4, 1, screenwidth * sizeof(float) * 4, iSiteTextureHeight, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaGraphicsUnmapResources(1, &grSite, 0));
/ *
	glReadBuffer(fbo_attachments[0]);
	//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);
	glReadPixels(1, 1, screenwidth, iSiteTextureHeight, GL_RGBA, GL_FLOAT, pReadBackValues);
* /


	f = 0;
	for (i=0; i<site_num; i++)
	{
		f += pReadBackValues[i * 4 + 2];
		g[i * 2] = 2 * pReadBackValues[i * 4]/ * /pReadBackValues[i*4+3]* /;
		g[i * 2 + 1] = 2 * pReadBackValues[i * 4 + 1]/ * /pReadBackValues[i*4+3]* /;
	}*/

	Energyf(grSite, g, f_tb_dev, screenwidth, iSiteTextureHeight, site_num, stream);

	lbfgsbcuda::CheckBuffer(g, site_num * 2, site_num * 2);
#ifdef DEBUG_TIME
	glFinish();
	get_timestamp(end_time);
	elapsed_time = (end_time-start_time);
	total_time += elapsed_time;
	printf("Step Scatter time: %f\n", elapsed_time);
#endif
#ifdef DEBUG_TIME
	get_timestamp(start_time);
#endif
	if (bShowTestResults)
	{
		///////////////////////////////////
		// Test pass, Display the result //
		///////////////////////////////////
		cgGLBindProgram(VP_FinalRender);
		cgGLBindProgram(FP_FinalRender);

		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[2], GL_RENDERBUFFER_EXT, RB_object);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[2]);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		if (FP_FinalRender_Size != NULL)
			cgSetParameter2f(FP_FinalRender_Size, screenwidth, screenheight);

		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);
		glActiveTextureARB(GL_TEXTURE2_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Color_Texture);

		glBegin(GL_QUADS);
			glVertex2f(1.0, 1.0);
			glVertex2f(1.0, float(screenheight+1));
			glVertex2f(float(screenwidth+1), float(screenheight+1));
			glVertex2f(float(screenwidth+1), 1.0);
		glEnd();

		nFuncCall++;
		if (bNewIteration)
		{
			bNewIteration = false;
			glReadBuffer(fbo_attachments[2]);
			//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

			//if (numIter==109)
			//{
			//	GLubyte *buffer_screen = new GLubyte[screenwidth*screenheight*4];
			//	char rawname[40];
			//	glReadPixels(1,1,screenwidth,screenheight,GL_RGBA,GL_UNSIGNED_BYTE,buffer_screen);
			//	strcpy(rawname, "a.raw");
			//	std::ofstream myrawfile(rawname, std::ios::binary);
			//	for (j=screenheight-1; j>=0; j--)
			//		for (i=0; i<screenwidth; i++)
			//		{
			//			myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4];
			//			myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4+1];
			//			myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4+2];
			//		}
			//	myrawfile.close();
			//	delete buffer_screen;
			//}
		}

		Current_Buffer = 1-Current_Buffer;
	}
	glFinish();
	get_timestamp(end_time_func);
	elapsed_time_func = (end_time_func-start_time_func);
	total_time_func += elapsed_time_func;

	f = *f_tb_host;
}

real BFGSOptimization()
{
	// Use L-BFGS method to compute new sites
	const real epsg = EPSG;
	const real epsf = EPSF;
	const real epsx = EPSX;
	const int maxits = MAXITS;
	stpscal = 2.75f; //Set for different problems!
	int info;

	total_time = 0;
	total_time_func = 0;

	real* x;
	int* nbd;
	real* l;
	real* u;
	memAlloc<real>(&x, site_num * 2);
	memAlloc<int>(&nbd, site_num * 2);
	memAlloc<real>(&l, site_num * 2);
	memAlloc<real>(&u, site_num * 2);
	memAllocHost<real>(&f_tb_host, &f_tb_dev, 1);

	InitSites(x, (float*)site_list_dev, sizeof(SiteType) / sizeof(float), nbd, l, u, site_num * 2, screenwidth);

	//lbfgsbcuda::CheckBuffer(x, site_num * 2, site_num * 2);
	/*
	real a1 = 1 / real(screenwidth - 1) * 2.0;
	real a2 = -a1 - 1.0;
	for (int i = 0; i < site_num; i++)
	{
		x[i * 2] = site_list[i].vertices[0].x * a1 + a2;
		x[i * 2 + 1] = site_list[i].vertices[0].y * a1 + a2;
		nbd[i * 2] = nbd[i * 2 + 1] = 2;
		l[i * 2] = l[i * 2 + 1] = -1.0;
		u[i * 2] = u[i * 2 + 1] = 1.0;
	}
*/

	printf("Start optimization...");
	get_timestamp(start_time);

	stpscal = 2.75f;

	int	m = 8;
	if (site_num * 2 < m)
		m = site_num * 2;
	for (int temp=0; temp<1; temp++)
	{
		bNewIteration = true;
		lbfgsbminimize(site_num*2, m, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
		//printf("Ending code:%d\n", info);
	}

	get_timestamp(end_time);
	elapsed_time = (end_time-start_time);
	total_time += elapsed_time;
	printf("Done.\n JFA Time: %lf\tBFGS Time: %lf\tTotal time: %lf\t", total_time_func, elapsed_time - total_time_func, elapsed_time);
	bReCompute = false;
	
	real f = DrawVoronoi(x);

	real* x_host = new real[site_num * 2];
	memCopy(x_host, x, site_num * 2 * sizeof(real), cudaMemcpyDeviceToHost);
	for(int i = 0; i < site_num; i++) {
		site_list[i].vertices[0].x = x_host[i * 2] * (screenwidth-1.0) + 1.0;
		site_list[i].vertices[0].y = x_host[i * 2 + 1] * (screenwidth-1.0) + 1.0;
	}
	delete[] x_host;

	memFreeHost(f_tb_host);
	memFree(x);
	memFree(nbd);
	memFree(l);
	memFree(u);

	return f;
}

real DrawVoronoi(real* xx)
{
	int i,j;

	real fEnergy = 1e20;

	GLfloat *buffer_screen = new GLfloat[screenwidth*screenheight*4];

	//FILE *site_file = fopen("result-uniform-CPU.txt", "r");

#ifdef DEBUG_TIME
	total_time = 0;
	get_timestamp(start_time);
#endif
	//////////////////////////////////////////////
	// First pass - Render the initial sites    //
	//////////////////////////////////////////////
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, FB_objects);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
		GL_TEXTURE_RECTANGLE_NV, Processed_Texture[0], 0);
	CheckFramebufferStatus();

	for (i=0; i<1; i++)
	{
		glDrawBuffer(fbo_attachments[i]);
		glClearColor(-1, -1, -1, -1);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glDrawBuffer(fbo_attachments[0]);

	cgGLEnableProfile(VertexProfile);
	cgGLEnableProfile(FragmentProfile);

	cgGLBindProgram(VP_DrawSites);
	cgGLBindProgram(FP_DrawSites);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(1, screenwidth+1, 1, screenheight+1);
	glViewport(1, 1, screenwidth, screenheight);

	DrawSites(false, xx, NULL);

	glReadBuffer(fbo_attachments[0]);
	//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

	Current_Buffer = 1;

#ifdef DEBUG_TIME
	get_timestamp(end_time);
	elapsed_time = (end_time-start_time);
	total_time += elapsed_time;
	printf("Step 1 time: %f\n", elapsed_time);
#endif

	bool Converged = false;
	while (!Converged)
	{
#ifdef DEBUG_TIME
		get_timestamp(start_time);
#endif
		/////////////////////////////////////
		// Second pass - Flood the sites   //
		/////////////////////////////////////
		cgGLBindProgram(VP_Flood);
		cgGLBindProgram(FP_Flood);

		if (VP_Flood_Size != NULL)
			cgSetParameter2f(VP_Flood_Size, screenwidth, screenheight);

		bool ExitLoop = false;
		bool SecondExit;
		int steplength;;
		SecondExit = (additional_passes==0);
		bool PassesBeforeJFA;
		PassesBeforeJFA = (additional_passes_before>0);
		if (PassesBeforeJFA)
			steplength = pow(2.0, (additional_passes_before-1));
		else
			steplength = (screenwidth>screenheight ? screenwidth : screenheight)/2;

		while (!ExitLoop)
		{
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[Current_Buffer], 
				GL_TEXTURE_RECTANGLE_NV, Processed_Texture[Current_Buffer], 0);
			CheckFramebufferStatus();
			glDrawBuffer(fbo_attachments[Current_Buffer]);

			glClearColor(-1, -1, -1, -1);
			glClear(GL_COLOR_BUFFER_BIT);

			//Bind & enable shadow map texture
			glActiveTextureARB(GL_TEXTURE0_ARB);
			glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);
			if (VP_Flood_Steplength != NULL)
				cgSetParameter1d(VP_Flood_Steplength, steplength);

			glBegin(GL_QUADS);
			glVertex2f(1.0, 1.0);
			glVertex2f(1.0, float(screenheight+1));
			glVertex2f(float(screenwidth+1), float(screenheight+1));
			glVertex2f(float(screenwidth+1), 1.0);
			glEnd();
			glReadBuffer(fbo_attachments[Current_Buffer]);
			//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

			if (steplength==1 && PassesBeforeJFA)
			{
				steplength = (screenwidth>screenheight ? screenwidth : screenheight)/2;
				PassesBeforeJFA = false;
			}
			else if (steplength>1)
				steplength /= 2;
			else if (SecondExit)
				ExitLoop = true;
			else
			{
				steplength = pow(2.0, (additional_passes-1));
				SecondExit = true;
			}
			Current_Buffer = 1-Current_Buffer;
		}
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);
		glReadPixels(1,1,screenwidth,screenheight,GL_RGBA,GL_FLOAT,buffer_screen);

#ifdef DEBUG_TIME
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time);
		total_time += elapsed_time;
		printf("Step 2 time: %f\n", elapsed_time);
#endif
		///////////////////////////////
		// Test pass, Compute energy //
		///////////////////////////////
		int Current_Energy_Buffer = 0;
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[0], 
			GL_TEXTURE_RECTANGLE_NV, Energy_Texture[Current_Energy_Buffer], 0);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[0]);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		cgGLBindProgram(VP_ComputeEnergyCentroid);
		cgGLBindProgram(FP_ComputeEnergyCentroid);

		if (FP_ComputeEnergyCentroid_Size != NULL)
			cgSetParameter2f(FP_ComputeEnergyCentroid_Size, screenwidth, screenheight);

		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, Processed_Texture[1-Current_Buffer]);

		glBegin(GL_QUADS);
		glVertex2f(1.0, 1.0);
		glVertex2f(float(screenwidth+1), 1.0);
		glVertex2f(float(screenwidth+1), float(screenheight+1));
		glVertex2f(1.0, float(screenheight+1));
		glEnd();

		glReadBuffer(fbo_attachments[0]);
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

		Current_Energy_Buffer = 1-Current_Energy_Buffer;

		//////////////////////
		// perform reduction
		//////////////////////
		cgGLBindProgram(VP_Deduction);
		cgGLBindProgram(FP_Deduction);

		bool ExitEnergyLoop = false;
		int quad_size = int((screenwidth>screenheight?screenwidth:screenheight)/2.0+0.5);
		while (!ExitEnergyLoop)
		{
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[0], 
				GL_TEXTURE_RECTANGLE_NV, Energy_Texture[Current_Energy_Buffer], 0);
			CheckFramebufferStatus();
			glDrawBuffer(fbo_attachments[0]);

			glClearColor(0, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT);

			//Bind & enable shadow map texture
			glActiveTextureARB(GL_TEXTURE0_ARB);
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB, Energy_Texture[1-Current_Energy_Buffer]);

			glBegin(GL_QUADS);
			glVertex2f(1.0, 1.0);
			glVertex2f(float(quad_size+1), 1.0);
			glVertex2f(float(quad_size+1), float(quad_size+1));
			glVertex2f(1.0, float(quad_size+1));
			glEnd();

			glReadBuffer(fbo_attachments[0]);
			//imdebugPixelsf(0, 0, GIMwidth+2, GIMheight+2, GL_RGBA);

			if (quad_size>1)
			{
				int temp = quad_size/2;
				quad_size = temp*2==quad_size ? temp : temp+1;
			}
			else
				ExitEnergyLoop = true;
			Current_Energy_Buffer = 1-Current_Energy_Buffer;
		}
		float total_sum[4];
		glReadBuffer(fbo_attachments[0]);
		glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, &total_sum);
		printf("Energy: %f\n", total_sum[0]);
		fEnergy = total_sum[0];

#ifdef DEBUG_TIME
		get_timestamp(start_time);
#endif
/*
		///////////////////////////////////
		// Test pass, Display the result //
		///////////////////////////////////
		cgGLBindProgram(VP_FinalRender);
		cgGLBindProgram(FP_FinalRender);

		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[2], GL_RENDERBUFFER_EXT, RB_object);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[2]);

		if (FP_FinalRender_Size != NULL)
			cgSetParameter2f(FP_FinalRender_Size, screenwidth, screenheight);

		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);
		glActiveTextureARB(GL_TEXTURE2_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Color_Texture);

		glBegin(GL_QUADS);
		glVertex2f(1.0, 1.0);
		glVertex2f(1.0, float(screenheight+1));
		glVertex2f(float(screenwidth+1), float(screenheight+1));
		glVertex2f(float(screenwidth+1), 1.0);
		glEnd();

		glReadBuffer(fbo_attachments[2]);*/
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);
/*
		GLubyte *buffer_screen = new GLubyte[screenwidth*screenheight*4];
		char rawname[40];
		glReadPixels(1,1,screenwidth,screenheight,GL_RGBA,GL_UNSIGNED_BYTE,buffer_screen);
		strcpy(rawname, "a.raw");
		std::ofstream myrawfile(rawname, std::ios::binary);
		for (j=screenheight-1; j>=0; j--)
			for (i=0; i<screenwidth; i++)
			{
				myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4];
				myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4+1];
				myrawfile << (unsigned char) buffer_screen[(j*screenwidth+i)*4+2];
			}
		myrawfile.close();
		delete buffer_screen;*/

#ifdef DEBUG_TIME
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time);
		total_time += elapsed_time;
		printf("Step test time: %f\n", elapsed_time);
#endif
#ifdef DEBUG_TIME
		get_timestamp(start_time);
#endif
		//////////////////////////////////////////
		// Third pass - Scatter points to sites //
		//////////////////////////////////////////
		cgGLBindProgram(VP_ScatterCentroid);
		cgGLBindProgram(FP_ScatterCentroid);

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[0], 
			GL_TEXTURE_RECTANGLE_NV, Site_Texture, 0);
		CheckFramebufferStatus();
		glDrawBuffer(buffers[0]);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		if (VP_ScatterCentroid_Size != NULL)
			cgSetParameter2f(VP_ScatterCentroid_Size, screenwidth, screenheight);

		//Bind & enable shadow map texture
		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1-Current_Buffer]);

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glBegin(GL_POINTS);
		for (i=0; i<screenwidth; i++)
			for (j=0; j<screenheight; j++)
				glVertex2f(i+1.5, j+1.5);
		glEnd();
		glDisable(GL_BLEND);

/*
		glReadBuffer(buffers[0]);
		float *tempTest = new float[screenwidth*10*4];
		glReadPixels(1, 1, screenwidth, 10, GL_RGBA, GL_FLOAT, tempTest);
		for (int index=0; index<site_num; index++)
		{
			if (tempTest[index*4+3]<=0)
			{
				imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);
				printf("Error! Empty VC found!\n");
				exit(0);
			}
		}*/

		Current_Buffer = 1-Current_Buffer;

#ifdef DEBUG_TIME
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time);
		total_time += elapsed_time;
		printf("Step 3 time: %f\n", elapsed_time);
#endif
#ifdef DEBUG_TIME
		get_timestamp(start_time);
#endif
		///////////////////////////////////////
		// Fourth pass - Test stop condition //
		///////////////////////////////////////
		cgGLBindProgram(VP_DrawSitesOQ);
		cgGLBindProgram(FP_DrawSitesOQ);

		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[2], GL_RENDERBUFFER_EXT, RB_object);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[2]);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		if (VP_DrawSitesOQ_Size != NULL)
			cgSetParameter2f(VP_DrawSitesOQ_Size, screenwidth, screenheight);

		//Bind & enable shadow map texture
		glActiveTextureARB(GL_TEXTURE0_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Site_Texture);
		glActiveTextureARB(GL_TEXTURE1_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[Current_Buffer]);

		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDepthMask(GL_FALSE);
		glBeginQueryARB(GL_SAMPLES_PASSED_ARB, occlusion_query);
		glBegin(GL_POINTS);
		for (i=0; i<site_num; i++)
		{
			float xx, yy;
			xx = i%screenwidth+1.5;
			yy = i/screenheight+1.5;
			glTexCoord1f(i);
			glVertex2f(xx, yy);
		}
		glEnd();
		glEndQueryARB(GL_SAMPLES_PASSED_ARB);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);

		glReadBuffer(fbo_attachments[2]);
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

		do{
			glGetQueryObjectivARB(occlusion_query, GL_QUERY_RESULT_AVAILABLE_ARB, &oq_available);
		}while(oq_available);
		glGetQueryObjectuivARB(occlusion_query, GL_QUERY_RESULT_ARB, &sampleCount);
		printf("sample count: %d\n", sampleCount);

		if (sampleCount==0)
			Converged = true;

		static int temp = 0;
		if (temp++>=0)
			Converged = true;

		if (Converged)
		{
#ifdef DEBUG_TIME
			printf("Total time: %f\n", total_time);
#endif
			//exit(0);
			break;
		}

#ifdef DEBUG_TIME
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time);
		total_time += elapsed_time;
		printf("Step 4 time: %f\n", elapsed_time);
#endif
#ifdef DEBUG_TIME
		get_timestamp(start_time);
#endif
		//////////////////////////////////////////////
		// Fifth pass - Draw sites at new positions //
		//////////////////////////////////////////////
		cgGLBindProgram(VP_DrawNewSites);
		cgGLBindProgram(FP_DrawNewSites);

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[Current_Buffer], 
			GL_TEXTURE_RECTANGLE_NV, Processed_Texture[Current_Buffer], 0);
		CheckFramebufferStatus();
		glDrawBuffer(fbo_attachments[Current_Buffer]);

		glClearColor(-1, -1, -1, -1);
		glClear(GL_COLOR_BUFFER_BIT);

		if (VP_DrawNewSites_Size != NULL)
			cgSetParameter2f(VP_DrawNewSites_Size, screenwidth, screenheight);

		glBegin(GL_POINTS);
		for (i=0; i<site_num; i++)
		{
			float xx, yy;
			xx = i%screenwidth+1.5;
			yy = i/screenheight+1.5;
			glTexCoord1f(i);
			glVertex2f(xx, yy);
		}
		glEnd();

		glReadBuffer(fbo_attachments[Current_Buffer]);
		//imdebugPixelsf(0, 0, screenwidth+2, screenheight+2, GL_RGBA);

		Current_Buffer = 1-Current_Buffer;
#ifdef DEBUG_TIME
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time);
		total_time += elapsed_time;
		printf("Step 5 time: %f\n", elapsed_time);
#endif
	}

	cgGLDisableProfile(VertexProfile);
	cgGLDisableProfile(FragmentProfile);

	////////////////////
	// compute measures
	////////////////////
	bool *bOnBoundary = new bool[site_num];
	bool *bIsHexagon = new bool[site_num];
	int *nNeighbors = new int[site_num*7];
	real *dDiameter = new real[site_num];
	real *dNeighborDist = new real[site_num];

	float site_pos[2], x, y, dist, neighbor_pos[2];
	int id, drow, dcol, nrow, ncol, neighbor_id, k;
	real dMaxDiameter, chi_id, chi;
	int nHex, nVC;

	for (id=0; id<site_num; id++)
	{
		bOnBoundary[id] = false;
		bIsHexagon[id] = true;
		nNeighbors[id*7] = 0;
		for (k=1; k<7; k++)
			nNeighbors[id*7+k] = -1;
		dDiameter[id] = -1;
		dNeighborDist[id] = 2*(screenwidth+screenheight);
	}
	dMaxDiameter = -1;
	chi = -1;
	nHex = nVC = 0;

	for (i=0; i<screenheight; i++)
	{
		for (j=0; j<screenwidth; j++)
		{
			site_pos[0] = buffer_screen[i*screenwidth*4+j*4];
			site_pos[1] = buffer_screen[i*screenwidth*4+j*4+1];
			id = int(buffer_screen[i*screenwidth*4+j*4+2]);
			x = j+1.5;
			y = i+1.5;
			site_pos[0] = (site_pos[0]-1)/screenwidth*2-1;
			site_pos[1] = (site_pos[1]-1)/screenheight*2-1;
			x = (x-1)/screenwidth*2-1;
			y = (y-1)/screenheight*2-1;
			dist = (x-site_pos[0])*(x-site_pos[0])+(y-site_pos[1])*(y-site_pos[1]);
			dist = sqrt(dist);
			dDiameter[id] = dDiameter[id]<dist ? dist : dDiameter[id];

			// traverse 9 neighbors
			for (drow=-1; drow<=1; drow++)
			{
				for (dcol=-1; dcol<=1; dcol++)
				{
					if (drow==0 && dcol==0)
						continue;
					nrow = i+drow;
					ncol = j+dcol;

					if (nrow<0 || nrow>=screenheight || ncol<0 || ncol>=screenwidth)
					{
						bOnBoundary[id] = true;
						continue;
					}

					neighbor_pos[0] = buffer_screen[nrow*screenwidth*4+ncol*4];
					neighbor_pos[1] = buffer_screen[nrow*screenwidth*4+ncol*4+1];
					neighbor_id = int(buffer_screen[nrow*screenwidth*4+ncol*4+2]);
					neighbor_pos[0] = (neighbor_pos[0]-1)/screenwidth*2-1;
					neighbor_pos[1] = (neighbor_pos[1]-1)/screenheight*2-1;
					if (neighbor_id==id)
						continue;

					dist = (neighbor_pos[0]-site_pos[0])*(neighbor_pos[0]-site_pos[0])
						   +(neighbor_pos[1]-site_pos[1])*(neighbor_pos[1]-site_pos[1]);
					dist = sqrt(dist);
					dNeighborDist[id] = dNeighborDist[id]>dist ? dist : dNeighborDist[id];

					for (k=1; k<7; k++)
					{
						if (nNeighbors[id*7+k]<0)
						{
							nNeighbors[id*7+k] = neighbor_id;
							nNeighbors[id*7]++;
							break;
						}
						else if (nNeighbors[id*7+k]==neighbor_id)
							break;
					}
					if (k==7)
						bIsHexagon[id] = false;
				}
			}
		}
	}
	for (id=0; id<site_num; id++)
	{
		if (nNeighbors[id*7]!=6)
			bIsHexagon[id] = false;
	}
	for (id=0; id<site_num; id++)
	{
		dMaxDiameter = dMaxDiameter<dDiameter[id] ? dDiameter[id] : dMaxDiameter;
		chi_id = 2*dDiameter[id]/dNeighborDist[id];
		chi = chi<chi_id ? chi_id : chi;
		if (!bOnBoundary[id])
		{
			nVC++;
		}
		if (bIsHexagon[id])
		{
			nHex++;
		}
	}

	printf("\n==== measures ====\n");
	printf("Number of VC in the middle: %d\n", nVC);
	printf("Number of hexagons: %d\n", nHex);
	printf("h: %f\n", dMaxDiameter);
	printf("chi: %f\n", chi);
	printf("==== measures ====\n\n");

#if 1
	GLubyte *ColorTexImage = new GLubyte[screenwidth*screenwidth*4];
	for (i=0; i<screenheight; i++)
	{
		for (j=0; j<screenwidth; j++)
		{
			int id = i*screenwidth+j;
			if (id<site_num)
			{
				if (bIsHexagon[id])
				{
					ColorTexImage[id*4] = 255;
					ColorTexImage[id*4+1] = 255; 
					ColorTexImage[id*4+2] = 255;
					ColorTexImage[id*4+3] = 255;
				}
				else
				{
					ColorTexImage[id*4] = 192;
					ColorTexImage[id*4+1] = 192; 
					ColorTexImage[id*4+2] = 192;
					ColorTexImage[id*4+3] = 255;
				}
			}
			else
			{
				ColorTexImage[id*4] = 
					ColorTexImage[id*4+1] = 
					ColorTexImage[id*4+2] = 
					ColorTexImage[id*4+3] = 0.0;
			}
		}
	}
	glActiveTextureARB(GL_TEXTURE2_ARB);
	glGenTextures(1, &Color_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Color_Texture);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, screenwidth,
		screenheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, ColorTexImage);

	delete ColorTexImage;
	//imdebugTexImagef(GL_TEXTURE_RECTANGLE_NV, Color_Texture, GL_R
#endif
	delete [] buffer_screen;
	delete [] bOnBoundary;
	delete [] bIsHexagon;
	delete [] nNeighbors;
	delete [] dDiameter;
	delete [] dNeighborDist;

	///////////////////////////////////
	// Last pass, Display the result //
	///////////////////////////////////
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);    
// 	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, fbo_attachments[2], GL_RENDERBUFFER_EXT, RB_object);
// 	CheckFramebufferStatus();
// 	glDrawBuffer(fbo_attachments[2]);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, screenwidth-1, 0, screenheight-1);
	glViewport(0, 0, screenwidth, screenheight);

	glActiveTextureARB(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[Current_Buffer]);
	glActiveTextureARB(GL_TEXTURE1_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Site_Texture);

	cgGLEnableProfile(VertexProfile);
	cgGLEnableProfile(FragmentProfile);

	cgGLBindProgram(VP_FinalRender);
	cgGLBindProgram(FP_FinalRender);

	// Set parameters of fragment program

	glBegin(GL_QUADS);
	glVertex2f(0.0, 0.0);
	glVertex2f(0.0, float(screenheight));
	glVertex2f(float(screenwidth), float(screenheight));
	glVertex2f(float(screenwidth), 0.0);
	glEnd();
/*	glReadBuffer(fbo_attachments[2]);*/
	//imdebugPixelsf(0, 0, screenwidth, screenheight, GL_RGBA);
// 	GLubyte *buffer_raw = new GLubyte[screenwidth*screenheight*4];
// 	glReadPixels(0,0,screenwidth,screenheight,GL_RGBA,GL_UNSIGNED_BYTE,buffer_raw);
// 	char rawname[40];
// 
// 	strcpy(rawname, "CVD.raw");
// 
// 	std::ofstream myrawfile(rawname, std::ios::binary);
// 	for (j=screenheight-1; j>=0; j--)
// 		for (i=0; i<screenwidth; i++)
// 		{
// 			myrawfile << (unsigned char) buffer_raw[(j*screenwidth+i)*4];
// 			myrawfile << (unsigned char) buffer_raw[(j*screenwidth+i)*4+1];
// 			myrawfile << (unsigned char) buffer_raw[(j*screenwidth+i)*4+2];
// 		}
// 	myrawfile.close();
// 	delete buffer_raw;

	cgGLDisableProfile(VertexProfile);
	cgGLDisableProfile(FragmentProfile);

/*
	if (site_visible)
	{
		DrawSites(true);
	}*/


	return fEnergy;
}

void Display(void)
{
	if (testFPS && frame_num == 0)
		get_timestamp(start_time);
	
	static real fEnergy = 1e20;
	static real fEnergyBase = 1e20;
	static bool isFirst = true;
	static real t0 = 0;
	static real tk = 0;

	static real k = 0;
	static real K = 1;

	while (bReCompute && k < K)
	{
		real fStar = BFGSOptimization();
		real df = fStar - fEnergy;
		if(df < 0) {
			//X <- X*
			CopySite(site_list_x, site_list, point_num);
			fEnergy = fStar;
			printf("Lower! e = %lf\n", fEnergy);
		} else {
			if(isFirst) {
				//initialize T0
				t0 = df * 4.48142;
				isFirst = false;
			}
			tk = t0 * pow(1.0 - k / K, 6);
			real acc = exp(-df / tk);
			real r = (float)rand() / (float)(RAND_MAX);
			if(r < acc) {
				//X <- X*
				CopySite(site_list_x, site_list, point_num);
				fEnergy = fStar;
				printf("Accepted! e = %lf, acc = %lf, tk = %lf\n", fEnergy, acc, tk);
			} else {
				printf("Rejected! e* = %lf > e = %lf\n", fStar, fEnergy);
			}
		}
		if(fStar < fEnergyBase) {
			//XBase <- X*
			CopySite(site_list_x_bar, site_list, point_num);
			fEnergyBase = fStar;
			FILE *site_file = fopen("sites_result.txt", "w");
			for(int i = 0; i < point_num; i++) {
				fprintf(site_file, "%f %f\n", (site_list[i].vertices[0].x - 1.0) / (screenwidth - 1.0), (site_list[i].vertices[0].y - 1.0) / (screenheight - 1.0));
			}
			fclose(site_file);
			printf("Base Updated!\n");
		}
		//Perturb X -> X*
		k = k + 1.0;
		for(int i = 0; i < point_num * 2; i++) {
			site_list_x[i] = site_list_x[i] + 
				((real)rand() / (real)RAND_MAX * 2.0 - 1.0) * 
				site_perturb_step * (real)screenwidth;
			site_list_x[i] = __max(1, __min(screenwidth - 1, site_list_x[i]));
		}
		CopySite(site_list, site_list_x, point_num);
		printf("* Energy Base = %lf *\n", fEnergyBase);
	}

	if(bReCompute) {
		CopySite(site_list, site_list_x_bar, point_num);
		real fStar = BFGSOptimization();
	}

	glFinish();

	frame_num++;
	if (testFPS && frame_num == 50)
	{
		frame_num = 0;
		get_timestamp(end_time);
		elapsed_time = (end_time-start_time)/50.0;
		printf("FPS: %f\n", 1000.0/elapsed_time);
	}

	glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y)
{
	int i;

	switch (key)
	{
	case '1':
		{
			mode = 1; // site = point
			printf("mode change to point.\n");
			break;
		}
	case '2':
		{
			mode = 2; // site = line
			printf("mode change to line.\n");
			break;
		}
	case '3':
		{
			mode = 3; // site = nurbs
			printf("mode change to NURBS.\n");
			break;
		}
	case '`':
		{
			site_visible = !site_visible;
			glutPostRedisplay();
			break;
		}
	case '.':
		{
			switch (mode)
			{
			case 1:
				{
					point_num++;
					break;
				}
			case 2:
				{
					line_num++;
					break;
				}
			case 3:
				{
					nurbs_num++;
					break;
				}
			}
			if (site_list)
			{
/*
				for (i=0; i<site_num; i++)
				{
					if (site_list[i].vertices)
						delete [] site_list[i].vertices;
				}*/
				cudaFreeHost(site_list);
			}
			site_num++;
			InitializeSites(point_num, line_num, nurbs_num);
			glutPostRedisplay();
			break;
		}
	case ',':
		{
			bool decreased = false;
			switch (mode)
			{
			case 1:
				{
					if (point_num>0)
					{
						point_num--;
						decreased = true;
					}
					break;
				}
			case 2:
				{
					if (line_num>0)
					{
						line_num--;
						decreased = true;
					}
					break;
				}
			case 3:
				{
					if (nurbs_num>0)
					{
						nurbs_num--;
						decreased = true;
					}
					break;
				}
			}
			if (decreased)
			{
				if (site_list)
				{
/*
					for (i=0; i<site_num; i++)
					{
						if (site_list[i].vertices)
							delete [] site_list[i].vertices;
					}*/
					cudaFreeHost(site_list);
				}
				site_num--;
				InitializeSites(point_num, line_num, nurbs_num);
				glutPostRedisplay();
			}
			break;
		}
	case 't':
		{
			frame_num = 0;
			testFPS = !testFPS;
			glutPostRedisplay();
			break;
		}
	case '\\':
		{
			animation = !animation;
			glutPostRedisplay();
			break;
		}
	case 's':
		{
			output = !output;
			glutPostRedisplay();
			break;
		}
	case 'q':
		{
			fclose(f_result);
			exit(0);
			break;
		}
	}
}

static void InitializeGlut(int *argc, char *argv[])
{
	int i,j;

	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(screenwidth, screenheight);
	glutCreateWindow(argv[0]);
	glutDisplayFunc(Display);
	glutKeyboardFunc(Keyboard);

	cudaSetDeviceFlags(cudaDeviceMapHost);

	cudaGLSetGLDevice(0);

	cublasCreate_v2(&cublasHd);

	glewInit();
	GLint max_texture_size;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);

	//Create the textures
	glActiveTextureARB(GL_TEXTURE0_ARB);

	glGenTextures(2, Processed_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[0]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA32F_ARB, screenwidth+2, screenheight+2, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Processed_Texture[1]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA32F_ARB, screenwidth+2, screenheight+2, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glGenTextures(1, &Site_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Site_Texture);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA32F_ARB, screenwidth+2, screenheight+2, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	cutilSafeCall(cudaGraphicsGLRegisterImage(&grSite, Site_Texture, 
		GL_TEXTURE_RECTANGLE_NV, cudaGraphicsMapFlagsReadOnly));

	glGenTextures(2, Energy_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Energy_Texture[0]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA32F_ARB, screenwidth+2, screenheight+2, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Energy_Texture[1]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA32F_ARB, screenwidth+2, screenheight+2, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glGenTextures(1, &IndexColor_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, IndexColor_Texture);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, screenwidth, screenheight, 0, 
		GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glGenFramebuffersEXT(1, &RB_object);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, RB_object);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA32F_ARB, screenwidth+2, screenheight+2);

	glGenFramebuffersEXT(1, &FB_objects);

	glGetQueryiv(GL_SAMPLES_PASSED_ARB, GL_QUERY_COUNTER_BITS_ARB, &oq_bitsSupported);
	glGenQueriesARB(1, &occlusion_query);

	InitCg();

	ScreenPointsList = glGenLists(1);
	glNewList(ScreenPointsList, GL_COMPILE);
	glBegin(GL_POINTS);
	for (i=0; i<screenwidth; i++)
		for (j=0; j<screenheight; j++)
			glVertex2f(i+1.5, j+1.5);
	glEnd();
	glEndList();

}

void CgErrorCallback(void)
{
	CGerror lastError = cgGetError();
	if (lastError)
	{
		printf("%s\n", cgGetErrorString(lastError));
		printf("%s\n", cgGetLastListing(Context));
		printf("Cg error, exiting...\n");
		exit(0);
	}
}

void InitializeSites(int point_num, int line_num, int nurbs_num)
{
	int i, j, index;
	int v_per_site;
	VertexSiteType v;

	memAllocHost<SiteType>(&site_list, &site_list_dev, point_num+line_num+nurbs_num);
	site_list_x = new float[(point_num+line_num+nurbs_num) * 2];
	site_list_x_bar = new float[(point_num+line_num+nurbs_num) * 2];
	site_perturb_step = 0.5f / sqrtf(point_num+line_num+nurbs_num);

// 	float *sites_array = new float[site_num*2];
// 	FILE *site_file = fopen("sites.txt", "r");
// 	//FILE *site_file = fopen("result-uniform-CPU.txt", "r");
// 	for (i=0; i<site_num; i++)
// 	{
// 		fscanf(site_file, "%f %f\n", &(sites_array[i*2]), &(sites_array[i*2+1]));
// 	}
// 	fclose(site_file);


	bool *FlagArray = new bool[screenwidth*screenwidth];
	for (i=0; i<screenwidth*screenheight; i++)
		FlagArray[i] = false;
	unsigned d = 123456;//GetTickCount();
	printf("Seed: %d\n", d);
	srand(d);
// 	const unsigned m = (unsigned)(ceilf(logf((point_num+line_num+nurbs_num) * 2) / logf(2.0f)));
// 	const Mcqmclfsr lfsr(m, Mcqmclfsr::GOOD_PROJECTIONS);
// 	const int scramble = rand() * (RAND_MAX + 1) + rand();
// 	unsigned state = 1 << (m - 1);
// 	const float org = lfsr.next(scramble, &state);

	for (i=0; i<point_num+line_num+nurbs_num; i++)
	{
		SiteType s;

		v.x = (double)rand()/(double)RAND_MAX*(screenwidth-1.0)+1.0;
		v.y = (double)rand()/(double)RAND_MAX*(screenheight-1.0)+1.0;
		while(true) {
			index = int(v.y)*screenwidth+int(v.x);

			if (FlagArray[index])
			{
				printf("\nDuplicate site found: #%d\n", i);
				//exit(0);
				v.x = v.x + ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * (float)(screenwidth-1);
				v.y = v.y + ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * (float)(screenwidth-1);

				while(v.x > (float)(screenwidth - 1)) {
					v.x -= (float)screenwidth;
				}

				while(v.x < 1.0f) {
					v.x += (float)screenwidth;
				}

				while(v.y > (float)(screenwidth - 1)) {
					v.y -= (float)screenwidth;
				}

				while(v.y < 1.0f) {
					v.y += (float)screenwidth;
				}
			}
			else
			{
				FlagArray[index] = true;
				break;
			}
		}
		s.vertices[0] = v;
		s.color[0] = (float)rand()/(float)RAND_MAX;
		s.color[1] = (float)rand()/(float)RAND_MAX;
		s.color[2] = (float)rand()/(float)RAND_MAX;
		s.color[3] = i;
		site_list[i] = s;
	}
	delete FlagArray;
/*	delete sites_array;*/
	GLubyte *ColorTexImage = new GLubyte[screenwidth*screenwidth*4];
	for (i=0; i<screenheight; i++)
		for (j=0; j<screenwidth; j++)
		{
			int id = i*screenwidth+j;
			if (id<point_num)
			{
				ColorTexImage[id*4] = site_list[id].color[0]*255;
				ColorTexImage[id*4+1] = site_list[id].color[1]*255; 
				ColorTexImage[id*4+2] = site_list[id].color[2]*255;
				ColorTexImage[id*4+3] = 255;
			}
			else
			{
				ColorTexImage[id*4] = 
				ColorTexImage[id*4+1] = 
				ColorTexImage[id*4+2] = 
				ColorTexImage[id*4+3] = 0.0;
			}
		}

	glActiveTextureARB(GL_TEXTURE2_ARB);
	glGenTextures(1, &Color_Texture);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, Color_Texture);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, screenwidth,
		screenheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, ColorTexImage);

	delete ColorTexImage;

	glGenBuffersARB(1, &vboId);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, point_num * sizeof(VertexSiteType), NULL, GL_DYNAMIC_DRAW_ARB);
	cudaGraphicsGLRegisterBuffer(&grVbo, vboId, cudaGraphicsMapFlagsWriteDiscard);


	glGenBuffersARB(1, &colorboId);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, colorboId);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, point_num * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	GLvoid* pointer = glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	float* sitelist = (float*)pointer;
	for (i=0; i<site_num; i++)
	{
		sitelist[i * 4 + 0] = site_list[i].color[0];
		sitelist[i * 4 + 1] = site_list[i].color[1];
		sitelist[i * 4 + 2] = site_list[i].color[2];
		sitelist[i * 4 + 3] = site_list[i].color[3];
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);

	//imdebugTexImagef(GL_TEXTURE_RECTANGLE_NV, Color_Texture, GL_RGBA);
}

void InitCg()
{
	cgSetErrorCallback(CgErrorCallback);
	Context = cgCreateContext();
	VertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
	cgGLSetOptimalOptions(VertexProfile);
	FragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(FragmentProfile);

	VP_DrawSites = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_DrawSites.cg",
							VertexProfile,
							NULL, NULL);
	FP_DrawSites = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_DrawSites.cg",
							FragmentProfile,
							NULL, NULL);
	VP_Flood = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_Flood.cg",
							VertexProfile,
							NULL, NULL);
	FP_Flood = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_Flood.cg",
							FragmentProfile,
							NULL, NULL);
	VP_Scatter = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_Scatter.cg",
							VertexProfile,
							NULL, NULL);
	FP_Scatter = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_Scatter.cg",
							FragmentProfile,
							NULL, NULL);
	VP_DrawNewSites = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_DrawNewSites.cg",
							VertexProfile,
							NULL, NULL);
	FP_DrawNewSites = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_DrawNewSites.cg",
							FragmentProfile,
							NULL, NULL);
	VP_DrawSitesOQ = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_DrawSitesOQ.cg",
							VertexProfile,
							NULL, NULL);
	FP_DrawSitesOQ = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_DrawSitesOQ.cg",
							FragmentProfile,
							NULL, NULL);
	VP_FinalRender = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_FinalRender.cg",
							VertexProfile,
							NULL, NULL);
	FP_FinalRender = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_FinalRender.cg",
							FragmentProfile,
							NULL, NULL);
	VP_ComputeEnergy = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_ComputeEnergy.cg",
							VertexProfile,
							NULL, NULL);
	FP_ComputeEnergy = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_ComputeEnergy.cg",
							FragmentProfile,
							NULL, NULL);
	VP_Deduction = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_Deduction.cg",
							VertexProfile,
							NULL, NULL);
	FP_Deduction = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_Deduction.cg",
							FragmentProfile,
							NULL, NULL);
	VP_ComputeEnergyCentroid = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_ComputeEnergyCentroid.cg",
							VertexProfile,
							NULL, NULL);
	FP_ComputeEnergyCentroid = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_ComputeEnergyCentroid.cg",
							FragmentProfile,
							NULL, NULL);
	VP_ScatterCentroid = cgCreateProgramFromFile(Context,
							CG_SOURCE, "VP_ScatterCentroid.cg",
							VertexProfile,
							NULL, NULL);
	FP_ScatterCentroid = cgCreateProgramFromFile(Context,
							CG_SOURCE, "FP_ScatterCentroid.cg",
							FragmentProfile,
							NULL, NULL);

	if(VP_DrawSites != NULL)
	{
		cgGLLoadProgram(VP_DrawSites);
	}
	if(FP_DrawSites != NULL)
	{
		cgGLLoadProgram(FP_DrawSites);
	}

	if(VP_Flood != NULL)
	{
		cgGLLoadProgram(VP_Flood);

		// Bind parameters to give access to variables in the shader
		VP_Flood_Steplength = cgGetNamedParameter(VP_Flood, "steplength");
		VP_Flood_Size = cgGetNamedParameter(VP_Flood, "size");
	}
	if(FP_Flood != NULL)
	{
		cgGLLoadProgram(FP_Flood);
	}

	if(VP_Scatter != NULL)
	{
		cgGLLoadProgram(VP_Scatter);

		// Bind parameters to give access to variables in the shader
		VP_Scatter_Size = cgGetNamedParameter(VP_Scatter, "size");
	}
	if(FP_Scatter != NULL)
	{
		cgGLLoadProgram(FP_Scatter);
	}

	if(VP_DrawNewSites != NULL)
	{
		cgGLLoadProgram(VP_DrawNewSites);

		// Bind parameters to give access to variables in the shader
		VP_DrawNewSites_Size = cgGetNamedParameter(VP_DrawNewSites, "size");
	}
	if(FP_DrawNewSites != NULL)
	{
		cgGLLoadProgram(FP_DrawNewSites);
	}

	if(VP_DrawSitesOQ != NULL)
	{
		cgGLLoadProgram(VP_DrawSitesOQ);

		// Bind parameters to give access to variables in the shader
		VP_DrawSitesOQ_Size = cgGetNamedParameter(VP_DrawSitesOQ, "size");
	}
	if(FP_DrawSitesOQ != NULL)
	{
		cgGLLoadProgram(FP_DrawSitesOQ);
	}

	if(VP_FinalRender != NULL)
	{
		cgGLLoadProgram(VP_FinalRender);
	}
	if(FP_FinalRender != NULL)
	{
		cgGLLoadProgram(FP_FinalRender);

		// Bind parameters to give access to variables in the shader
		FP_FinalRender_Size = cgGetNamedParameter(FP_FinalRender, "size");
	}

	if(VP_ComputeEnergy != NULL)
	{
		cgGLLoadProgram(VP_ComputeEnergy);
	}
	if(FP_ComputeEnergy != NULL)
	{
		cgGLLoadProgram(FP_ComputeEnergy);

		// Bind parameters to give access to variables in the shader
		FP_ComputeEnergy_Size = cgGetNamedParameter(FP_ComputeEnergy, "size");
	}

	if(VP_Deduction != NULL)
	{
		cgGLLoadProgram(VP_Deduction);
	}
	if(FP_Deduction != NULL)
	{
		cgGLLoadProgram(FP_Deduction);
	}

	if(VP_ComputeEnergyCentroid != NULL)
	{
		cgGLLoadProgram(VP_ComputeEnergyCentroid);
	}
	if(FP_ComputeEnergyCentroid != NULL)
	{
		cgGLLoadProgram(FP_ComputeEnergyCentroid);

		// Bind parameters to give access to variables in the shader
		FP_ComputeEnergyCentroid_Size = cgGetNamedParameter(FP_ComputeEnergyCentroid, "size");
	}

	if(VP_ScatterCentroid != NULL)
	{
		cgGLLoadProgram(VP_ScatterCentroid);

		// Bind parameters to give access to variables in the shader
		VP_ScatterCentroid_Size = cgGetNamedParameter(VP_ScatterCentroid, "size");
	}
	if(FP_ScatterCentroid != NULL)
	{
		cgGLLoadProgram(FP_ScatterCentroid);
	}
}

int main(int argc, char *argv[])
{
	point_num = 8000;
	screenwidth = screenheight = 1024;	
	if (argc==1) {
		printf("Point NUM#: \n");
		scanf("%d", &point_num);
		printf("Resolution: \n");
		scanf("%d", &screenwidth);
		int qn = 0;
		while(screenwidth != 0) {
			qn++;
			screenwidth >>= 1;
		}
		screenwidth = 1 << (qn - 1);
		screenheight = screenwidth;
	} else
		point_num = atoi(argv[1]);

	if(screenwidth <= 1 || point_num < 2) {
		printf("Invalid Args!\n");
		return -1;
	}

	if (argc==3)
		bShowTestResults = atoi(argv[2]);

	line_num = 0;
	nurbs_num = 0;
	site_num = point_num + line_num + nurbs_num;
	mode = 1; // mode = point sites
	speed = 3.0f;
	additional_passes = 0;
	additional_passes_before = 0;
	bReCompute = true;

//	screenwidth = screenheight = 2048;
	InitializeGlut(&argc, argv);

	srand((unsigned)time(NULL));
	srand(34143214); // standard random seed
	//srand(8149040);
	controlpoints = (float *)malloc(sizeof(float)*12);
	InitializeSites(point_num, line_num, nurbs_num);

	iSiteTextureHeight = site_num/screenwidth+1;
	pReadBackValues = new float[screenwidth*iSiteTextureHeight*4];

	glutMainLoop();
	
	glDeleteBuffersARB(1, &vboId);
	glDeleteBuffersARB(1, &colorboId);
	cudaGraphicsUnregisterResource(grSite);
	cudaGraphicsUnregisterResource(grVbo);

	cgDestroyProgram(VP_DrawSites);
	cgDestroyProgram(FP_DrawSites);
	cgDestroyProgram(VP_Flood);
	cgDestroyProgram(FP_Flood);
	cgDestroyProgram(VP_FinalRender);
	cgDestroyProgram(FP_FinalRender);
	cgDestroyContext(Context);
	
	delete[] site_list_x;
	delete[] site_list_x_bar;
	cudaFreeHost(site_list);

	cublasDestroy_v2(cublasHd);

	cudaDeviceReset();

	return 0;
}