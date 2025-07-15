import os
from typing import Any, List
from openpyxl import Workbook, load_workbook


def write_excel(path: str, sheet_name: str, data: List[List[Any]]) -> bool:
    """
    Write data to excel's assigned sheet
    
    Inputs
    --------
        path (str): 
            Excel path to write data.
            If exist file, over-write it.
        sheet_name (str): 
            The sheet name to write data.
            If exist sheet, will add suffix to create a new one.
        data (list[list[any]]): 
            Write datas, sub-list is a line.
    """

    try:
        if os.path.exists(path):
            wb = load_workbook(path)
        else:
            wb = Workbook()
            default_sheet = wb.active
            wb.remove(default_sheet)
        
        original_sheet_name = sheet_name
        count = 1
        
        while sheet_name in wb.sheetnames:
            sheet_name = f"{original_sheet_name}_{count}"
            count += 1

        ws = wb.create_sheet(title=sheet_name)

        for row in data:
            ws.append(row)

        wb.save(path)
        return True
    except Exception as e:
        print(str(e))
        return False