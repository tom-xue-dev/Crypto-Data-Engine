from services.tick_data_scraper.worker import submit_download

def register_tasks(celery_app):

    @celery_app.task(name="tick.download")
    def dispatch_tick_download(cfg: dict):
        return submit_download(cfg)
