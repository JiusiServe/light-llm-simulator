---
name: email-writer
description: Help compose professional simulation result emails. Use this skill when the user asks to "write email", "compose email", "draft email", "write a message to " or mentions sending an email about simulation results.
---

# Email Writer

This skill helps you compose simulation result emails with a specific format and send them via Gmail API.

## Email Configuration

- **Sender**: yujuancao07@gmail.com
- **Receivers**:
  - liuhongsheng4@huawei.com
  - jiangchenzhou@huawei.com
  - chuyu1@huawei.com
  - caoyujuan1@huawei.com
- **Subject**: YYYY/MM/DD simulation results (using UTC+8 time when email is sent)
- **Data Source**: https://drive.google.com/drive/folders/17EJN3WPwGO4D-31q6rCNZySFlWDU3uHJ
- **Throughput Google Drive**: https://drive.google.com/drive/folders/1mRdt8LNCdPUuRkd-SNcKdsBm-84MXJWO

## Gmail API Configuration

- **GMAIL_API_URL**: https://script.google.com/macros/s/AKfycbwtlpgjhrgjROvMgDufP_XCYEWPvVAv-vgPfqtAlJZqHldmgQNcAu4kj4w8wsstIRXkLQ/exec
- **GMAIL_API_TOKEN**: e95689b7cbae3e3f9a9aa082194ebd98

Note: URL-encode the subject and body parameters.

## Email Format

Use plain text format with section labels like **【Setting】**, **【Results】**, **【TODO】** instead of markdown headers (##).

## Default Setting

```
【Setting】

- Model: DeepSeekV3
- Hardware:
  - DV120/DV100异构
  - DV120/A2异构
  - DV100/A2异构
- 总die数:
  - ffn: 32, attn: 32, 64, 96
  - ffn: 144, attn: 144, 288
- 通信: 层次化通信，AFD场景不支持两段式通信
- 量化:
  - 权重: int8
  - 激活: int8
  - KVCache: bf16
  - 注: vllm-ascend不支持KVCache int8量化，海思已支持KVCache int8量化，影响mla_prolog算子、FusedInferAttentionScore算子
- TPOP: 20ms, 50ms, 70ms, 100ms, 150ms
- kv_len: 2k, 4k, 8k, 16k, 128k
- AFD micro batch num: 2, 3
- MTP: 1
- Parallelism Strategy: attn DP, ffn EP
```

## Default TODO

```
【TODO】

固定FFN die数，不断增大attn die数，进行性能摸高：
1. EP32, DV120/A2异构，性能摸高
2. EP144, DV120/DV100异构，性能摸高
```

## Workflow

When writing an email, follow this process:

### Step 1: Setting
1. Show the default Setting section
2. **Ask user to confirm** or modify before proceeding

### Step 2: Results
1. Find images in `data/images/throughput/` and subfolders (EP32, EP144, etc.)
2. Filter images based on user's Setting configuration
3. List the images that will be included
4. Write the Results section (describe images/attachments)
5. **Ask user to confirm** before proceeding

### Step 3: Upload Images
After confirming Results section, upload images to Google Drive:
```bash
python .claude/skills/visualization/upload_throughput.py
```

Then get individual Google Drive links for each uploaded image and include them in the email body.

### Step 4: TODO
1. Show the default TODO section
2. **Ask user to confirm** or modify before proceeding

### Step 5: Final Review
1. Show complete email draft with individual image links
2. Ask user for final approval

### Step 6: Send Email
After user approves, send the email using Gmail API:

1. URL-encode the subject and body
2. Run the curl command with receivers:

```bash
curl -sL "https://script.google.com/macros/s/AKfycbwtlpgjhrgjROvMgDufP_XCYEWPvVAv-vgPfqtAlJZqHldmgQNcAu4kj4w8wsstIRXkLQ/exec?token=e95689b7cbae3e3f9a9aa082194ebd98&action=send&to=RECEIVERS&subject=ENCODED_SUBJECT&body=ENCODED_BODY"
```

3. Confirm email was sent successfully

Note: Replace `RECEIVERS` with comma-separated email addresses.
