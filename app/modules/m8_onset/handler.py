from fastapi import APIRouter, Depends

from app.modules.m1_intake.service import load
from app.modules.m8_onset.service import onset_report
from app.store import Store, get_store

router = APIRouter(prefix='/api/onset', tags=['起点对齐'])


@router.get('/{source_id}')
def get_onset(source_id: str, store: Store = Depends(get_store)):
    return onset_report(load(store, source_id))
