//------------------------------------------------------------------------------------------
//	Includes
//------------------------------------------------------------------------------------------
#include "App.h"
#include "FPS.h"

//------------------------------------------------------------------------------------------
//	constructor & destructor
//------------------------------------------------------------------------------------------
App::App()
{
	SecureZeroMemory(this, sizeof(App));	//クラス内のすべての変数をゼロ初期化
}
App::~App()
{
}


//------------------------------------------------------------------------------------------
//	アプリケーションの実行
//------------------------------------------------------------------------------------------
int App::Run()
{
	if (FAILED(InitWindow()))	//Initialize the window
	{
		return 0;
	}

	if (FAILED(InitD3D()))	//Initialize Direct3D
	{
		return 0;
	}

	if (FAILED(InitShader()))	//Initialize Shader
	{
		return 0;
	}

	MainLoop();	//メインループへ

	return 0;	//アプリ終了
}


//------------------------------------------------------------------------------------------
//	ウィンドウプロシージャー
//------------------------------------------------------------------------------------------
LRESULT CALLBACK App::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_KEYDOWN:
		switch ((char)wParam)
		{
		case VK_ESCAPE:	//ESCキーで修了
			PostQuitMessage(0);
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}

	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}


//------------------------------------------------------------------------------------------
//	ウィンドウ作成
//------------------------------------------------------------------------------------------
bool App::InitWindow()
{
	// インスタンスハンドルを取得.
	HINSTANCE hInst = GetModuleHandle(nullptr);
	if (!hInst)
	{
		return FALSE;
	}

	// ウィンドウの定義
	WNDCLASSEX  wc;
	SecureZeroMemory(&wc, sizeof(wc));
	wc.cbSize			= sizeof(WNDCLASSEX);					// 構造体の大きさ
	wc.style			= CS_HREDRAW | CS_VREDRAW;				// スタイル 
	wc.lpfnWndProc		= WndProc;								// メッセージ処理関数
	wc.hInstance		= hInst;								// プログラムのハンドル
	wc.hIcon			= LoadIcon(nullptr, IDI_APPLICATION);	// アイコン
	wc.hCursor			= LoadCursor(nullptr, IDC_ARROW);		// カーソル
	wc.hbrBackground	= (HBRUSH)GetStockObject(LTGRAY_BRUSH);	// ブラシ
	wc.lpszMenuName		= nullptr;								// メニュー
	wc.lpszClassName	= APP_NAME;								// クラス名
	wc.hIconSm			= LoadIcon(nullptr, IDI_APPLICATION);

	RegisterClassEx(&wc);	//ウインドウクラスを登録

	// ウィンドウの作成
	m_hWnd = CreateWindow(
		APP_NAME,				// ウインドウクラス名 
		APP_NAME,				// ウインドウのタイトル
		WS_OVERLAPPEDWINDOW,	// ウインドウスタイル 
		200, 200,				// ウインドウ表示位置 
		WINDOW_WIDTH, 			// ウインドウの大きさ 
		WINDOW_HEIGHT,			// ウインドウの大きさ 
		nullptr,				// 親ウインドウのハンドル 
		nullptr,				// メニューのハンドル 
		hInst,					// インスタンスのハンドル
		nullptr);				// 作成時の引数保存用ポインタ 

	// エラーチェック
	if (!m_hWnd)
	{
		return E_FAIL;
	}

	//ウインドウの表示
	ShowWindow(m_hWnd, SW_SHOW);
	UpdateWindow(m_hWnd);

	return true;
}


//------------------------------------------------------------------------------------------
//	Initializes Direct3D
//------------------------------------------------------------------------------------------
bool App::InitD3D()
{
	HRESULT hr = S_OK;

	// デバイスとスワップチェーンの作成
	DXGI_SWAP_CHAIN_DESC sd;
	SecureZeroMemory(&sd, sizeof(sd)); //初期化
	sd.BufferCount							= 1;
	sd.BufferDesc.Width						= WINDOW_WIDTH;						//サイズ指定
	sd.BufferDesc.Height					= WINDOW_HEIGHT;					//サイズ指定
	sd.BufferDesc.Format					= DXGI_FORMAT_R8G8B8A8_UNORM;		//32bit color
	sd.BufferDesc.RefreshRate.Numerator		= 60;								//リフレッシュレート
	sd.BufferDesc.RefreshRate.Denominator	= 1;								//バックバッファの数
	sd.BufferUsage							= DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow							= m_hWnd;
	sd.SampleDesc.Count						= 1;
	sd.SampleDesc.Quality					= 0;
	sd.Windowed								= TRUE;								//Fullscreen表示の指定


	D3D_FEATURE_LEVEL pFeatureLevels = D3D_FEATURE_LEVEL_11_0;
	
	hr = D3D11CreateDeviceAndSwapChain(
			nullptr,
			D3D_DRIVER_TYPE_HARDWARE,
			nullptr,
			0,
			&pFeatureLevels,
			1,
			D3D11_SDK_VERSION,
			&sd,
			m_pSwapChain.GetAddressOf(),
			m_pDevice.GetAddressOf(),
			nullptr,
			&m_pDeviceContext);

	if (FAILED(hr))
	{
		return false;
	}


	//レンダーターゲットビューの作成
	ComPtr<ID3D11Texture2D> pBackBuffer;	//受け皿
	m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)pBackBuffer.GetAddressOf());
	m_pDevice->CreateRenderTargetView(pBackBuffer.Get(), nullptr, m_pRenderTargetView.GetAddressOf());


	//深度ステンシルビューの作成
	D3D11_TEXTURE2D_DESC descDepth;
	descDepth.Width					= WINDOW_WIDTH;
	descDepth.Height				= WINDOW_HEIGHT;
	descDepth.MipLevels				= 1;
	descDepth.ArraySize				= 1;
	descDepth.Format				= DXGI_FORMAT_D32_FLOAT;
	descDepth.SampleDesc.Count		= 1;
	descDepth.SampleDesc.Quality	= 0;
	descDepth.Usage					= D3D11_USAGE_DEFAULT;
	descDepth.BindFlags				= D3D11_BIND_DEPTH_STENCIL;
	descDepth.CPUAccessFlags		= 0;
	descDepth.MiscFlags				= 0;

	m_pDevice->CreateTexture2D(&descDepth, nullptr, m_pDepthStencil.GetAddressOf());
	m_pDevice->CreateDepthStencilView(m_pDepthStencil.Get(), nullptr, m_pDepthStencilView.GetAddressOf());


	//ラスタライズ設定
	D3D11_RASTERIZER_DESC rdc;
	SecureZeroMemory(&rdc, sizeof(rdc));
	rdc.CullMode = D3D11_CULL_NONE;		//裏面も塗りつぶす　
	rdc.FillMode = D3D11_FILL_SOLID;	//ポリゴン内部も塗りつぶす
	rdc.FrontCounterClockwise = TRUE;	//時計回りに頂点を結ぶ（三角形メッシュ）

	m_pDevice->CreateRasterizerState(&rdc, &m_pRasterizerState);
	m_pDeviceContext->RSSetState(m_pRasterizerState.Get());


	return true;
}


//------------------------------------------------------------------------------------------
//	Initialize Shader
//------------------------------------------------------------------------------------------
bool App::InitShader()
{
	HRESULT hr = S_OK;
	ComPtr<ID3DBlob> pCompiledShader = nullptr;
	ComPtr<ID3DBlob> pErrors = nullptr;

	UINT Flag1 = D3D10_SHADER_PACK_MATRIX_COLUMN_MAJOR | D3D10_SHADER_ENABLE_STRICTNESS;
#if defined(DEBUG) || defined(_DEBUG)
	Flag1 |= D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION | D3D10_SHADER_OPTIMIZATION_LEVEL0;
#else
	Flag1 |= D3D10_SHADER_OPTIMIZATION_LEVEL3;
#endif

	//ブロブからDRAW_GLINTバーテックスシェーダー作成
	hr = D3DX11CompileFromFile(
			L"Resources/Shaders/DrawTexture.hlsl",
			nullptr,
			nullptr,
			"VS",
			"vs_5_0",
			Flag1,
			0,
			nullptr,
			pCompiledShader.GetAddressOf(),
			pErrors.GetAddressOf(),
			nullptr);

	if (FAILED(hr))
	{
		MessageBoxA(m_hWnd, (LPSTR)pErrors->GetBufferPointer(), nullptr, MB_OK);
		return false;
	}

	hr = m_pDevice->CreateVertexShader(
			pCompiledShader->GetBufferPointer(),
			pCompiledShader->GetBufferSize(),
			nullptr,
			m_pVertexShader[DRAW_GLINT].GetAddressOf());

	if (FAILED(hr))
	{
		return false;
	}


	//DRAW_GLINTシェーダー用　頂点インプットレイアウトを作成
	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	int numElements = sizeof(layout) / sizeof(layout[0]);

	hr = m_pDevice->CreateInputLayout(
			layout,
			numElements,
			pCompiledShader->GetBufferPointer(),
			pCompiledShader->GetBufferSize(),
			m_pVertexLayout[DRAW_GLINT].GetAddressOf());

	if (FAILED(hr))
	{
		return false;
	}


	//ブロブからDRAW_GLINTピクセルシェーダー作成
	hr = D3DX11CompileFromFile(
			L"Resources/Shaders/DrawTexture.hlsl",
			nullptr,
			nullptr,
			"PS",
			"ps_5_0",
			Flag1,
			0,
			nullptr,
			pCompiledShader.GetAddressOf(),
			pErrors.GetAddressOf(),
			nullptr);

	if (FAILED(hr))
	{
		MessageBoxA(m_hWnd, (LPSTR)pErrors->GetBufferPointer(), nullptr, MB_OK);
		return false;
	}

	hr = m_pDevice->CreatePixelShader(
			pCompiledShader->GetBufferPointer(),
			pCompiledShader->GetBufferSize(),
			nullptr,
			m_pPixelShader[DRAW_GLINT].GetAddressOf());

	if (FAILED(hr))
	{
		return false;
	}


	//DRAW_GLINTシェーダー用　ポリゴンバーテックスバッファー作成
	Vertex vertices[] =
	{
		D3DXVECTOR3(-1.0, -1.0, 0),	//頂点1
		D3DXVECTOR3(-1.0, 1.0, 0),	//頂点2
		D3DXVECTOR3(1.0, -1.0, 0),	//頂点3
		D3DXVECTOR3(1.0, 1.0, 0)	//頂点4
	};
	D3D11_BUFFER_DESC bd;
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Vertex) * 4;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	InitData.pSysMem = vertices;

	hr = m_pDevice->CreateBuffer(&bd, &InitData, m_pVertexBuffer.GetAddressOf());

	if (FAILED(hr))
	{
		return false;
	}


	//コンスタントバッファー作成
	D3D11_BUFFER_DESC cbDesc;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.ByteWidth = sizeof(ConstantBuffer);
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;

	hr = m_pDevice->CreateBuffer(&cbDesc, nullptr, &m_pConstantBuffer[DRAW_GLINT]);

	if (FAILED(hr))
	{
		return false;
	}


	return true;
}

//------------------------------------------------------------------------------------------
//	アプリケーション処理。アプリのメイン関数。
//------------------------------------------------------------------------------------------
void App::MainLoop()
{
	// メッセージループ
	MSG msg = { 0 };
	SecureZeroMemory(&msg, sizeof(msg));

	FPS fps;

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			//render
			OnRender();
			fps.COUNTER(m_hWnd);
		}
	}
	
}


//------------------------------------------------------------------------------------------
//	描画処理
//------------------------------------------------------------------------------------------
void App::OnRender()
{
	//ビュー行列
	D3DXVECTOR3 cameraPos(0.0f, 0.0f, -2.42f);	//カメラ位置
	D3DXVECTOR3 lookAtPos(0.0f, 0.0f, 0.0f);	//注視位置
	D3DXVECTOR3 upVec(0.0f, 1.0f, 0.0f);		//上方位置
	D3DXMatrixLookAtLH(&m_ViewMatrix, &cameraPos, &lookAtPos, &upVec);


	//プロジェクション行列（射影変換）
	D3DXMatrixPerspectiveFovLH(
		&m_ProjectionMatrix,
		(float)D3DX_PI / 4.0,							//視野角
		(float)WINDOW_WIDTH / (float)WINDOW_HEIGHT,		//アスペクト比
		0.1f,											//near clip
		100.0f);										//far clip

	float ClearColor[4] = { 0.178, 0.178, 0.178, 1 };// クリア色作成　RGBAの順


	//ビューポートの設定
	D3D11_VIEWPORT vp;
	vp.Width	= WINDOW_WIDTH;
	vp.Height	= WINDOW_HEIGHT;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;

	m_pDeviceContext->RSSetViewports(1, &vp);


	//レンダーターゲットビューと深度ステンシルビューをパイプラインにバインド
	m_pDeviceContext->OMSetRenderTargets(1, m_pRenderTargetView.GetAddressOf(), m_pDepthStencilView.Get());

	m_pDeviceContext->ClearRenderTargetView(m_pRenderTargetView.Get(), ClearColor);	//画面クリア
	m_pDeviceContext->ClearDepthStencilView(m_pDepthStencilView.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);	//深度バッファクリア


	//バーテックスバッファーをセット
	UINT stride = sizeof(Vertex);
	UINT offset = 0;
	m_pDeviceContext->IASetVertexBuffers(0, 1, m_pVertexBuffer.GetAddressOf(), &stride, &offset);


	//シェーダーのコンスタントバッファーに各種データを渡す
	ConstantBuffer cb;
	D3D11_MAPPED_SUBRESOURCE pData;

	static float startTime = timeGetTime();
	float currentTime = timeGetTime() - startTime;
	
	POINT point;
	GetCursorPos(&point);
	ScreenToClient(m_hWnd, &point);

	if (SUCCEEDED(m_pDeviceContext->Map(m_pConstantBuffer[DRAW_GLINT].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &pData)))
	{
		//WVP行列をシェーダーに渡す
		cb.WVP = m_ViewMatrix*m_ProjectionMatrix;
		//D3DXMatrixTranspose(&cb.mWP, &cb.mWP);

		cb.time = currentTime / 1000.0f;

		cb.mouse = D3DXVECTOR2(point.x, point.y);

		memcpy_s(pData.pData, pData.RowPitch, (void*)(&cb), sizeof(cb));
		m_pDeviceContext->Unmap(m_pConstantBuffer[DRAW_GLINT].Get(), 0);
	}

	
	//このコンスタントバッファーを使うシェーダーの登録
	m_pDeviceContext->VSSetConstantBuffers(0, 1, m_pConstantBuffer[DRAW_GLINT].GetAddressOf());
	m_pDeviceContext->PSSetConstantBuffers(0, 1, m_pConstantBuffer[DRAW_GLINT].GetAddressOf());


	//頂点インプットレイアウトをセット
	m_pDeviceContext->IASetInputLayout(m_pVertexLayout[DRAW_GLINT].Get());
	
	
	//プリミティブ・トポロジーをセット
	m_pDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	//使用するシェーダーの登録	
	m_pDeviceContext->VSSetShader(m_pVertexShader[DRAW_GLINT].Get(), nullptr, 0);
	m_pDeviceContext->PSSetShader(m_pPixelShader[DRAW_GLINT].Get(), nullptr, 0);
	
	
	//プリミティブをレンダリング
	m_pDeviceContext->Draw(4, 0);

	m_pSwapChain->Present(0, 0);	//画面更新（バックバッファをフロントバッファに）	
}