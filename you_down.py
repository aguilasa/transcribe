from yt_dlp import YoutubeDL
import os

def baixar_audio_youtube_avancado(url, formato='mp3', qualidade='192', pasta_destino='downloads'):
    """
    Baixa o áudio de um vídeo do YouTube com opções avançadas

    Args:
        url: URL do vídeo do YouTube
        formato: Formato do áudio (mp3, m4a, wav, etc)
        qualidade: Qualidade do áudio em kbps
        pasta_destino: Pasta onde o arquivo será salvo
    """
    # Cria a pasta de destino se não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': formato,
            'preferredquality': qualidade,
        }],
        'outtmpl': os.path.join(pasta_destino, '%(title)s.%(ext)s'),
        'verbose': True,
        'quiet': False,
        'no_warnings': False,
        # Opções adicionais
        'noplaylist': True,  # Não baixa playlists
        'extract_flat': False,
        'writethumbnail': False,  # Não baixa thumbnail
        'writesubtitles': False,  # Não baixa legendas
        'writedescription': False,  # Não baixa descrição
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Primeiro extrai as informações do vídeo
            info = ydl.extract_info(url, download=False)
            video_title = info['title']
            duration = info.get('duration', 0)  # Duração em segundos

            print(f"Iniciando download de: {video_title}")
            print(f"Duração: {duration//60}:{duration%60:02d}")

            # Realiza o download
            ydl.download([url])

            arquivo_saida = os.path.join(pasta_destino, f"{video_title}.{formato}")
            print(f"Download concluído: {arquivo_saida}")
            return arquivo_saida

    except Exception as e:
        print(f"Erro ao baixar o vídeo: {str(e)}")
        return None

# Exemplo de uso
url_video = "https://www.youtube.com/live/p95VYdLmbHY"
arquivo = baixar_audio_youtube_avancado(
    url=url_video,
    formato='mp3',
    qualidade='192',
    pasta_destino='downloads'
)

# Created/Modified files during execution:
# ["downloads/{titulo_do_video}.mp3"]