import io
from typing import Union
from fastapi import FastAPI, Response, status, UploadFile

import local_utils
from inference_cpu import inference
import uuid
import aiofiles
import zipfile

app = FastAPI()


@app.get("/ai_exec")
def get_ai_rtss_result(filename: str, response: Response):

    try:
        inference(i_filename=filename)
        return {
            "isError": False,
            "message": "CheXNet with localization AI execution successful"
        }
    except Exception as e:
        print(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "isError": True,
            "message": e
        }
    pass


pass


@app.post("/ai_exec")
async def post_ai_exec(file: Union[UploadFile, None] = None, response: Response = None):
    if not file:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "isError": True,
            "message": "No upload file sent"
        }
    else:
        output_filename = f"temp/{uuid.uuid4().hex}.dcm"
        async with aiofiles.open(output_filename, "wb") as output_file:
            upload_file_content = await file.read()
            await output_file.write(upload_file_content)
        pass
        try:
            gsps_obj_list = inference(i_filename=output_filename)
            s = io.BytesIO()
            zip_file = zipfile.ZipFile(s, "w")

            for gsps_obj in gsps_obj_list:
                gsps_filename = gsps_obj["filename"]
                zip_file.write(f"temp/{gsps_filename}", gsps_filename)
            pass
            zip_file.close()
            file_response = Response(s.getvalue(),
                                     media_type="application/x-zip-compressed",
                                     headers={
                                         'Content-Disposition': f'attachment;filename={uuid.uuid4().hex}.zip'
                                     })
            return file_response
        except Exception as e:
            print(e)
            return {
                "isError": True,
                "message": e
            }
        pass
    pass
pass


@app.post("/ai_exec_multipart")
async def post_ai_exec_multipart(file: Union[UploadFile, None] = None, response: Response = None):
    if not file:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "isError": True,
            "message": "No upload file sent"
        }
    else:
        output_filename = f"temp/{uuid.uuid4().hex}.dcm"
        async with aiofiles.open(output_filename, "wb") as output_file:
            upload_file_content = await file.read()
            await output_file.write(upload_file_content)
        pass
        try:
            gsps_obj_list = inference(i_filename=output_filename)
            bytes_arr = bytearray()
            boundary = f"--{uuid.uuid4()}-{uuid.uuid4()}"
            content_type = "Content-Type: application/dicom\r\n\r\n"
            for gsps_obj in gsps_obj_list:
                gsps_dataset = gsps_obj["dataset"]
                gsps_bytes = local_utils.write_dataset_to_bytes(gsps_dataset)
                bytes_arr.extend(boundary.encode())
                bytes_arr.extend("\r\n".encode())
                bytes_arr.extend(content_type.encode())
                bytes_arr.extend(gsps_bytes)
                bytes_arr.extend("\r\n".encode())
            pass
            bytes_arr.extend(f"{boundary}--".encode())
            file_response = Response(content=bytes(bytes_arr),
                                     headers={
                                         "content-type": f"multipart/related; type=\"application/dicom\"; boundary={boundary}"
                                     })
            return file_response
        except Exception as e:
            print(e)
            return {
                "isError": True,
                "message": e
            }
        pass
    pass
pass




