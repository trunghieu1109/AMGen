from common import get_json_response_from_gpt, get_json_response_from_gpt_reflect, _pack_message
from prompts.abstract_based_prompt import INTERACTION_PATTERN, ABSTRACTED_WORKFLOW_TEMPLATE
import asyncio
import copy

async def high_level_task_decomposition(meta_model, query, n_repeat):
    potential_high_level_plan = []
    
    for i in range(0, n_repeat):
        task_high_level_decomposition = f"""
Role and Responsibility:
You are an expert LLM assistant trained to analyze and decompose a user query into its core subtasks. Your goal is to:

- Identify and list essential subtasks that are necessary to accomplish the user’s input query.
- Each subtask must represent a distinct and meaningful action or process step that contributes directly to solving the overall task, do not analyze too specific to each details in query.
- These subtasks must follow the reasoning model, from subtasks 1 -> subtask 2.
- Remove many trivial steps that are not the core subtasks.
- Maximum 4 subtasks.
User query:
{query}

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "subtask_list": [
        {{
            "objective": "Brief and clear description of the core subtask"
        }}
    ]
}}

Each objective must be a concise, clear, and self-contained description of a core subtask.

Do not include any additional fields or explanations outside of the objective key.
    """
        msg_list = [
            {"role": "user", "content": task_high_level_decomposition},
        ]

        high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['subtask_list'], 0.0)
        print(f"\n=========== high level decomposition, attempt {i} ==========\n", high_level_decomposition)
        potential_high_level_plan.append(high_level_decomposition)
        
    final_high_level_decomposition_prompt = f"""
I have a set of candidate plans, each representing a possible solution to the same task. Please analyze all the given plans and select the most popular and best one based on the following criteria:
1. Popularity: If possible, consider how frequently each plan (or similar structure/idea) appears among the options.
2. Effectiveness: Which plan is most likely to succeed in accomplishing the intended task?
3. Clarity and feasibility: Is the plan clear, implementable, and realistic?

Potential High Level Plan:
{potential_high_level_plan}

Maximum 4 subtasks.

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "subtask_list": [
        {{
            "objective": "Brief and clear description of the core subtask"
        }}
    ]
}}

Each objective must be a concise, clear, and self-contained description of a core subtask.

Do not include any additional fields or explanations outside of the objective key.
    """
    
    msg_list = [
        {"role": "user", "content": final_high_level_decomposition_prompt},
    ]

    final_high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'subtask_list'], 0.0)
    
    return final_high_level_decomposition['subtask_list']

async def task_analysis(meta_model, query):
    task_detail_analysis = f"""
Provide a detailed analysis of the given problem by addressing the following points. Ensure the response is clear, logical, and uses appropriate mathematical notation. The analysis should be thorough yet concise, avoiding unnecessary computation unless requested.
Do not specify the objective / goal of this question.
1. Extract and Summarize Given Information:
    - Identify all key details provided in the problem, including numerical values, geometric objects, conditions, constraints.
    - Summarize the properties of any mathematical or geometric entities involved (e.g., shapes, dimensions, equations, or parameters).
2. Analyze Relationships Between Components:
    - Describe how the given entities or quantities are interconnected (e.g., geometric constraints, algebraic relationships, dependencies between variables).
    - Explain the significance of any conditions or constraints (e.g., tangency, equality, bounds) and how they influence the problem’s structure.
    - Hypothesize how the components might contribute to solving the problem or achieving the stated objective.
3. Identify the Field of Study:
    - Specify the mathematical domain(s) relevant to the problem (e.g., geometry, algebra, calculus, number theory, combinatorics).
    - Mention any subfields or specific concepts involved (e.g., 3D geometry, differential equations, graph theory).
    - Note potential applications or contexts where such problems arise (e.g., mathematical competitions, physics, computer science).
4. Highlight Aspects Needing Clarification:
    - Identify any ambiguous terms, conditions, or assumptions in the problem statement (e.g., undefined positions, unclear constraints).
    - List potential challenges in interpreting or solving the problem (e.g., complex computations, multiple possible configurations).

Note:
- Do not answer or suggest solution to solve the query. Only analysis information in it.
- Only extract the information from this query. 

User query:
{query}

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "analysis": ""
}}

Do not include any additional fields or explanations outside of the objective key.
    """
    
    msg_list = [
        {"role": "user", "content": task_detail_analysis},
    ]

    detailed_analysis ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'analysis'], 0.0)
    
    return detailed_analysis['analysis']

async def specific_task_decomposition(meta_model, query, collaboration_dependencies=None, aw_flow = None, prev_task_decomposition = None, evaluation = None, is_refinement = False, potential_op={}):
    flow_range = {}
    
    cnt = -1
    stage_cnt = 0
    st = []
    flow_name = []
    ed = 0
    for item in aw_flow:
        if item.startswith("Stage"):
            ed = int(item[-1])
            stage_cnt += 1
        else:
            if aw_flow[item]['flow_type'].startswith('start'):
                cnt += 1 
                st.append(stage_cnt)
                flow_name.append(aw_flow[item]['flow_type'])
            else:
                # customized_task_decomposition['control_flow'][cnt]['range'] = [f'stage_{st[cnt]}', f'stage_{ed}']
                flow_range[flow_name[cnt]] = [f'stage_{st[cnt]}', f'stage_{ed}']
                cnt -= 1
                
    if not is_refinement:       
        user_prompt = f"""
You are an agent specializing in task decomposition. Given a query, break it into stages and control flows, with each stage containing subtasks aligned with the provided abstract workflow, dependencies, and control flow structure.

**\[Inputs]**

* Query: {query}
* Abstract Workflow: {aw_flow}
* Control Flow Range: {flow_range} (stage_id of start and end stages)
* Potential Agent Collaboration Pattern: {potential_op}

**\[Instructions]**

1. Decompose the query into a plan based on the Abstract Workflow, with up to 5 subtasks total across all stages, avoiding trivial divisions.
2. The plan includes:
   2.1. **Stages**: Sets of subtasks with a shared role, each a dictionary with:
      * `stage_id`: e.g., `stage_x`
      * `title`: Common task of subtasks
      * `subtasks`: Array of dictionaries, each with:
        * `id`: Format `stage_x.subtask_y` (x, y start from 1)
        * `objective`: Clear, actionable instruction
        * `dependencies`: Array of subtask IDs (e.g., `stage_x.subtask_y`, `previous.stage_x.subtask_y` for prior loop iterations) that must be completed first. Ensure:
          - Dependencies use outputs from prior subtasks/stages/iterations.
          - Include dependencies from outside to inside loops, if applicable.
          - Within loops, allow dependencies on any subtask in the same loop (current or prior iterations, using `previous.stage_x.subtask_y`).
        * `agent_collaboration`: Collaboration pattern from provided options.
   2.2. **Control Flows**: Define stage execution order:
      * `control_flow_type`: `sequential` or `loop`
      * `iteration_criteria`: For loops, specify repetition (default: `"2 iterations"` if unspecified)
      * `range`: `[start_stage_id, end_stage_id]` (same ID for single stage)
3. Strictly adhere to the **Abstract Workflow** structure:
   3.1. Map all control flows (including loop flows) into the task decomposition, ensuring:
     * Correct order of stages and control flows.
     * Inclusion of loop flows as specified, even if the query does not inherently require looping, nesting stages inside the loop.
     * Default `iteration_criteria` (e.g., `"2 iterations"`) if not specified.
   3.2. Each stage may contain multiple subtasks contributing to the same general task.
   3.3. Define dependencies when one subtask relies on another’s output, including:
     * Dependencies from subtasks outside the loop to those inside.
     * Dependencies within a loop, where a subtask could depend on any subtask in the same loop (current or previous iterations). Use `previous.stage_x.subtask_y` for prior iterations.
     * One subtask could depend on a subtask from a previous iteration (if applicable) to enable iterative refinement.
**\[Example]**
For a query calculating a physical quantity iteratively, suppose the Abstract Workflow specifies a loop over `stage_0` (compute quantity) and `stage_1` (evaluate results) for 2 iterations. The decomposition might include:
- `stage_0.subtask_1`: Compute initial value, depends on `previous.stage_1.subtask_1` (evaluation from prior iteration).
- `stage_0.subtask_2`: Refine computation, depends on `stage_0.subtask_1` and `previous.stage_1.subtask_1` (evaluation from prior iteration).
- `stage_1.subtask_1`: Evaluate accuracy, depends on `stage_0.subtask_2`.
This ensures `stage_0.subtask_1` and `stage_0.subtask_2` in iteration 2 uses `previous.stage_1.subtask_1` from iteration 1 to refine its computation.
4. Ensure subtasks use outputs from prior steps/iterations to support others, creating dependencies where applicable, including within loops.
5. Align subtasks with query details and Abstract Workflow.

**\[Output Format — JSON]**

```json
{{
  "thought": "<Brief reasoning on query breakdown, control flow interpretation, loop dependencies (e.g., previous.stage_x.subtask_y), and past feedback>",
  "task_decomposition": {{
    "stages": [
      {{
        "stage_id": "<e.g., stage_y>",
        "title": "<Stage purpose>",
        "subtasks": [
          {{
            "id": "<e.g., stage_x.subtask_y>",
            "objective": "<Clear instruction>",
            "dependencies": ["<e.g., stage_x.subtask_y or previous.stage_x.subtask_y>", ...],
            "agent_collaboration": "<Collaboration pattern>"
          }},
          ...
        ]
      }},
      ...
    ],
    "control_flow": [
      {{
        "control_flow_type": "sequential | loop",
        "iteration_criteria": "<e.g., '2 iterations'>",
        "range": ["<start_stage_id>", "<end_stage_id>"]
      }},
      ...
    ]
  }}
}}
```

**Additional Notes:**

- Maximum 5 subtasks total across all stages. Do not decompose too deeply.
- Dependencies within loops can reference any subtask in the same loop (current or prior iterations, using `previous.stage_x.subtask_y`).
- Subtasks cannot depend on themselves, even in prior iterations.
- Concretize steps using query details, ensuring dependencies reflect output usage.
"""
    else:
        failure_reason = [eva.get("failure_reason", "") for eva in evaluation]
        user_prompt = f"""
**You are an agent specializing in task decomposition refinement.** The previous task decomposition encountered several issues that led to inefficiency. Based on the evaluation and feedback, your task is to **revise the previous decomposition** with improved structure, better dependencies, and well-justified subtasks and control flows.

---

### **\[Inputs]**

* **Query**: {query}
* **Previous Decomposition**: {prev_task_decomposition}
* **Evaluation Feedback**: {evaluation}
* **Dependencies and Agent Collaboration Info**: {collaboration_dependencies}
* **Potential Agent Collaboration Patterns**: {potential_op}
* Control Flow Range (the stage_id of the starting and ending stages of a control flow. All stages between and including these belong to the control flow): {flow_range}
---

### **\[Instructions]**

1. ** Do not decompose query to too small / trivial subtasks **. Ensure each subtask is specific for the query, do not too abstract.
2. **Leverage feedback from the previous attempt** to:
   * Add missing dependencies.
   * Clarify ambiguous subtasks.
   * Split overly complex subtasks.
   * Improve the logical order and information propagation between subtasks.
   * Explicitly incorporate **failure reasons and expert feedback into objectives** of the respective subtasks.
   * Add những feedback về lỗi trong qua trình reasoning vào trong objective của subtasks để tránh lặp lại lỗi sai. Nguyên nhân bị sai: {failure_reason}

5. **Subtasks must use context appropriately**:
   * Dependencies are not limited to immediately preceding subtasks—subtasks can draw on context from any earlier relevant subtask.
   * Ensure no subtask produces unused output—**every output should feed into another subtask**, either directly or indirectly.

6. Ensure the decomposition is:
   * **More granular**, but **not trivial**.
   * **Logically flowing** and **reasoning-aligned**.
   * Aligned with the agent collaboration patterns based on complexity.
   
7. The execution flow must strictly follow the definitions in the Control Flow Range.
   - Nếu không có điều chỉnh gì về control flow, giữ nguyên so với `previous task decomposition`.
---

**\[Output format — JSON]**
{{
  "thought": "<Brief reasoning on how the query was broken down, including how control flows and loop dependencies (especially previous.stage_x.subtask_y) were interpreted, and how past feedback was addressed.>",
  "task_decomposition": {{
    "stages": [
      {{
        "stage_id": "<e.g., stage_y>",
        "title": "<Stage title from abstract workflow>",
        "subtasks": [
          {{
            "id": "<e.g., stage_x.subtask_y>",
            "objective": "<Clear instruction for the subtask>",
            "dependencies": ["<e.g., stage_x.subtask_y or previous.stage_x.subtask_y>", ...],
            "agent_collaboration": "<Suitable agent collaboration pattern>"
          }},
          ...
        ]
      }},
      ...
    ],
    "control_flow": [
      {{
        "control_flow_type": "sequential | loop",
        "iteration_criteria": "<For loop: e.g., '2 iterations' or 'until convergence'>",
        "range": ["<start_stage_id>", "<end_stage_id>"]
      }},
      ...
    ]
  }}
}}

Additional Note:
* Tránh lặp lại sai lầm từ lần trước bằng cách trích dẫn lý do thất bại trong từng subtask.
* Nếu `evaluation` có đề cập vấn đề “thiếu gắn kết giữa các bước”, thì dependencies phải được cải thiện triệt để.
* Nếu một subtask trong lần trước “không tạo ra đầu ra hữu ích”, thì hãy loại bỏ hoặc điều chỉnh mục tiêu của nó.
* Còn lại cố gắng giữ nguyên bản chất của plan ban đầu, chỉ thay đổi những phần nhận được phản hồi và cách để khắc phục.
        """

    msg_list = [
        {"role": "user", "content": user_prompt},
    ]

    task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'task_decomposition'], 0.0)
    customized_task_decomposition = task_decomposition['task_decomposition']
    
    for stage_idx, stage in enumerate(customized_task_decomposition['stages']):
        for subtask_idx, subtask in enumerate(stage['subtasks']):
            deps = customized_task_decomposition['stages'][stage_idx]['subtasks'][subtask_idx]['dependencies']
            customized_task_decomposition['stages'][stage_idx]['subtasks'][subtask_idx]['dependencies'] = []
            deps_set = set()
            if len(deps) > 0:
                for dep in deps:
                    pos = dep.find("stage")
                    if dep[pos:] == subtask['id']:
                        continue
                    
                    if dep[pos:] not in deps_set:
                        customized_task_decomposition['stages'][stage_idx]['subtasks'][subtask_idx]['dependencies'].append(dep)
                    
                    deps_set.add(dep[pos:])    
                    
    for stage_idx, stage in enumerate(customized_task_decomposition['stages']):
        for subtask_idx, subtask in enumerate(stage['subtasks']):
            deps = customized_task_decomposition['stages'][stage_idx]['subtasks'][subtask_idx]['dependencies']
            if len(deps) > 0:
                customized_task_decomposition['stages'][stage_idx]['subtasks'][subtask_idx]['objective'] += "\nInput content are results (both thinking and answer) from: " + " & ".join([st_dep.replace("previous.", "former iterations of ") for st_dep in deps]) + ", respectively."
    task_decomposition['task_decomposition'] = customized_task_decomposition
    print("\n =================== Planner Thought ================\n", task_decomposition['thought'])
    return task_decomposition['task_decomposition']
    
async def generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, query, potential_op={}):
    user_prompt_generate_workflow = f"""
You are tasked with generating a concrete multi-stage agentic workflow using techniques like Chain-of-Thought, Self-Consistency, Reflexion, or Debate. Your goal is to implement the workflow for the provided query, following the task decomposition and improving from previous feedback.

Ensure this code is runnable. Double-check many times.

[Inputs]
1. Query: {query}
2. Task Decomposition: {task_decomposition}
3. Abstract Workflow Template: {ABSTRACTED_WORKFLOW_TEMPLATE}. Return code in format of a function `forward`
4. Interaction Pattern Guide: {interaction_pattern}
5. Potential Agent Collaboration Pattern (You only choose pattern from this list): {potential_op}.

[Requirements]
1. Format your output in JSON format as:
{{
  "thought": "Brief reasoning behind your workflow implementation.",
  "code": "Python code implementing the concrete agentic workflow."
}}
2. For each subtask:
   - Ensure that each subtask strictly follows the instruction defined in the task decomposition and incorporates relevant evaluation feedback and suggestions. Thêm cả phần mô tả input trong instruction (giống như trong task decomposition) để bổ trợ cho việc reasoning.
   - Apply the exactly agent collaboration patterns to each subtask as specified in task decomposition.
   - Pass the outputs from Subtask A to Subtask B in accordance with the defined dependencies between the subtasks.
   - Follow the task decomposition strictly (stage, order, dependencies).
   - Implement using appropriate agent collaboration pattern (CoT, SC-CoT, Reflexion, Debate).
   - Replace any placeholder with task-specific content grounded in the query.
   
3. For each control flow:
   - range defines the stage_id of the starting and ending stages of the control flow. All stages between (and including) these will be grouped together.
   - Sequential control flow: Execute the subtasks in the defined range of stages sequentially.
   - Loop control flow:
        + Group all subtasks from the stages defined in the range, and execute them using a for loop. The iteration_criterias field defines the loop condition or the number of iterations.
        + You must maintain a dictionary to store the results of the subtasks in this control flow so they can be used later.
        + If a subtask A has a dependency on another subtask B that is located inside a loop control flow, it means that the current subtask A requires access to all outputs produced by that inner subtask B across every iteration of the loop, not just the latest one.
        + Đối với dependencies dạng previous của các subtask trong loop, điều này tức là tất cả kết quả của subtasks đó (cả thinking và answer) từ các iterations trước đó sẽ được sử dụng làm input.
Ex:
-> Wrong cases: Only get the latest iteration results of subtask 1.
```python
loop_results = {{}} # save results of iterations.

for i in range(0, n_iteration):
    results1, logs1 = await self.cot(...) # subtask 1
    loop_results[<subtask_id of subtask 1>]['answer'].append(results1['answer'])
    loop_results[<subtask_id of subtask 1>]['thinking'].append(results1['thinking'])
    
    cot_agent_desc_2 = {{
        "instruction": "..."
        "input": [taskInfo, results1['thinking'], results1['answer']],
        ...
    }}
    results2, logs2 = await self.cot(..., cot_agent_desc=cot_agent_desc_2, ....) # subtask 2
```

-> Correct cases: Use the results from all previous iterations instead of the latest one of the subtask 1.
```python
loop_results = {{}} # save results of iterations.

for i in range(0, n_iteration):
    results1, logs1 = await self.cot(...) # subtask 1
    loop_results[<subtask_id of subtask 1>]['answer'].append(results1['answer'])
    loop_results[<subtask_id of subtask 1>]['thinking'].append(results1['thinking'])
    
    cot_agent_desc_2 = {{
        "instruction": "..."
        "input": [taskInfo] + loop_results[<subtask_id of subtask 1>]['answer'] + loop_results[<subtask_id of subtask 1>]['thinking'], 
        ...
    }}
    results2, logs2 = await self.cot(..., cot_agent_desc=cot_agent_desc_2, ....) # subtask 2
```

4. Preserve logic from abstract workflow:
    - Use same number of stages.
    - Link subtasks clearly via outputs/inputs.
    - Synthesize final answer using `make_final_answer`.
   
6. Do not use LaTeX.
7. Do not include comments in code.
8. Ensure the final_answer generate by `self.make_final_answer`.
Eg. 
```python
final_answer = await self.make_final_answer(resultsn['thinking'], resultsn['answer'])
return final_answer, logs
```
9. Do not write instruction in possessive form.

[Goal]
Ensure all subtasks are well-structured, grounded, and correctly executed. Output a fully working Python-based workflow.
Do not use `\` to break line.

    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_generate_workflow},
    ]
    next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'code'], 0.0)
    print("================ Generated Multi-agent system ================\n", next_solution['code'])
    # next_solution['code'] = next_solution['code'].replace("{{", "{")
    # next_solution['code'] = next_solution['code'].replace("}}", "}")
    
    return next_solution
    
async def evaluate_workflow(query, subtask_desc, current_ans, output_description, verifier_hub):
    reasoning_refinement_prompt = f"""
You are a Verification Agent. Your task is to review the reasoning process of previous agents to find errors, check if contexts were sufficient, analyze how agents interacted, and suggest workflow improvements. The final answer is known to be wrong based on professor feedback — identify why and how to fix it.

# Input
1. User Query: {query}
2. Subtasks:
   - Instruction
   - Context
   - Response (thinking + answer)
   - Final Answer: {current_ans}
3. Output Format: {output_description}
4. Reasoning Process of Previous Agents:
{subtask_desc}

# Tasks
1. Reasons of Failure:
   - Phân tích một cách chi tiết lý do tại sao mà reasoning process trước đó bị sai? Sai ở bước nào? Lý do tại sao lại sai?
    - Mô tả một cách rất chi tiết lỗi sai.
2. **Reasoning Check**:
   - Spot flawed logic or wrong assumptions in subtask reasoning.
   - Identify where the error happened that caused the final answer to fail.
   - Xác định cụ thể lỗi sai trong reasoning process (sai ở điểm suy luận nào đâu? tại sao lại sai? nên sửa như thế nào? sửa bằng cách nào?).
   - Nghiêm cấm đưa ra đáp án trực tiếp của câu hỏi trong feedback. 
   - Phân tích xem lỗi sai bắt nguồn từ bước nào?

3. **Context Evaluation**:
   - Was context for each subtask enough and relevant?
   - If not, what was missing or unclear?
   - Xem xét trong quá trình reasoning của previous agents và code của workflow. Suy nghĩ xem liệu rằng có thể đưa thêm context từ bước A -> bước B nào đó hay không?
   
4. **Agent Interaction Analysis**:
   - Did subtasks pass outputs correctly?
   - Was the collaboration pattern (e.g. CoT, Debate, Reflexion, SC CoT) effective?
   - Did poor interaction contribute to failure?
   - Do subtasks need to change the collaboration patterns? 
   - Tại các subtasks bị thất bại, xem xét việc chuyển đối agent collaboration patterns của subtasks sang một patterns mạnh mẽ hơn.

5. **Workflow Improvement**:
Base on feedback about reasoning process, propose potential improvements if necessary, such as:
    - Refine or break down subtasks that were failed.
    - Use better collaboration patterns for failed subtasks.
    - Improve instructions of failed subtasks.
    - Reconnect context between steps.
Only return from 1 - 2 core solutions, which directly affect to the reasoning process.

# Output
Return JSON with:
{{  
    "failure_reason": "The reason for why previous process was failed."
    "feedback": "Your detailed analysis: what failed, where, and why.",
    "suggestion": "Concrete steps to improve the workflow based on feedback"
}}

# Notes
- Be clear and concise.
- Call out correct subtasks too.
- Prioritize root cause fixes.
    """
    
    msg_list = [
        {"role": "user", "content": reasoning_refinement_prompt},
    ]
    
    tasks = [
        (verifier_model, asyncio.create_task(
            get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback', 'failure_reason'], 0.0)
        ))
        for verifier_model in verifier_hub
    ]
    
    # Run all tasks in parallel and gather results
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    evaluation = []
    # Process results in order of verifiers
    for (verifier_model, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"================ Error for {verifier_model} ================\n", str(result))
            continue
        mas_feedback, _ = result
        evaluation.append({
            "verifier_name": verifier_model,
            "evaluation": mas_feedback
        })
        
    for ev in evaluation:
        print(f"\n================= {ev['verifier_name']} ==================\n")
        print(ev["evaluation"])
    
    return evaluation

async def merge_filtered_workflow(meta_model, filtered_sequential_only_workflow, filtered_loop_contain_workflow, filtered_conditional_contain_workflow=""):
    sequential_dep = []
    sequential_op = []
    sequential_flow = ""
    if filtered_sequential_only_workflow:
        for idx, subtask in enumerate(filtered_sequential_only_workflow['specific_workflow']):
            if 'dependencies' in subtask:
                for dep in subtask['dependencies']:
                    sequential_dep.append(f"{dep.replace("subtask", "stage")} has dependency on {subtask['subtask_id'].replace("subtask", "stage")}")
                    
                sequential_op.append(f"{dep.replace("subtask", "stage")} usually implemented by {subtask['agent_collaboration']} patterns")
        sequential_flow = str(filtered_sequential_only_workflow['flow']).replace("subtask", "stage")
            
    loop_dep = []
    loop_op = []
    loop_flow = ""
    if filtered_loop_contain_workflow:
        for idx, subtask in enumerate(filtered_loop_contain_workflow['specific_workflow']):
            if 'dependencies' in subtask:
                for dep in subtask['dependencies']:
                    loop_dep.append(f"{dep.replace("subtask", "stage")} has dependency on {subtask['subtask_id'].replace("subtask", "stage")}")
            
                loop_op.append(f"{dep.replace("subtask", "stage")} usually implemented by {subtask['agent_collaboration']} patterns")
                
        
        loop_flow = str(filtered_loop_contain_workflow['flow']).replace("subtask", "stage")
    
    print("Sequential flow: \n", sequential_flow)
    print("Loop flow: \n", loop_flow)
    
    print("Sequential dep: \n", sequential_dep)
    print("Loop dep: \n", loop_dep)
    
    print("Sequential op: \n", sequential_op)
    print("Loop op: \n", loop_op)
    
    user_prompt_merge_workflow = f"""
You are a specialized AI agent for analyzing and merging task workflows represented in structured JSON format. Your task is to intelligently merge two similar but not identical workflows into a unified, optimized version that preserves both shared and unique stages, while maintaining logical structure and relative control flow positions.

### 🔧 INPUT

You will be given **two workflows**, each represented as a list of stages in JSON format. Each stage has the following fields:
- `stage_id`: unique identifier
- `stage_name`: human-readable name
- `abstracted_objective`: a concise description of what the stage accomplishes
- `agent_collaboration`: either a string (e.g., "logic code") or a list of collaboration methods (e.g., `["CoT", "AnswerGenerate"]`)
- `dependencies` (optional): list of stage_id that this stage depends on.

Each workflow may also contain **control flow markers**:
- `start_sequential` / `end_sequential`
- `start_loop` / `end_loop`
- `start_conditional` / `end_conditional`
- `start_true_branch` / `end_true_branch`
- `start_false_branch` / `end_false_branchs`

You will be provided with:
- Filtered Sequential Only Workflow:
{sequential_flow}
Dependencies between stages:
{sequential_dep}
Agent collaboration pattern:
{sequential_op}

- Filtered Loop Contain Workflow:
{loop_flow}
Dependencies between stages:
{loop_dep}
Agent collaboration pattern:
{loop_op}

### 📌 TASK

1. **Merge the two workflows** into a single workflow list that:
   - Keeps the relative position of control flows (`start_sequential`, `start_loop`, etc.) **exactly as they appear** in each respective flow.
   - If one or a group of stages serves a similar purpose as those in the remaining workflow, merge them together (compare their overall abstracted objectives). Avoid duplicating or redundantly repeating stages that have already been performed.
Example: stage 1 and stage 2 from workflow A are semantically similar to stage 1 in workflow B, then merge them together -> stage 1 and stage 2 in the merged_workflow.
   - For distinct stages that appear only in one workflow, consider including them in the merged workflow if they are not redundant and logically fit within the flow.
   - Adjust `dependencies` accordingly in the merged version.

2. **Ensure logical consistency**: 
   - Maintain correct ordering of dependencies.

3. **Avoid**:
   - Breaking control flow structure (e.g., nesting start/end incorrectly).
   - Duplicate identical or redundant stages.
   - Losing important agent collaboration roles or objectives.
   - Blindly appending all stages together.

### 🧠 Reasoning Strategy

While merging, follow these reasoning steps:
- Align control flow structure first.
- Identify matching stages by comparing their names, objectives, collaboration patterns, and structural position.
- Merge identical stages, keep unique ones if relevant.
- Double-check the merged workflow, make sure that it follow instructions correctly. Then refine to make final result.

### 🧾 OUTPUT FORMAT
{{
    "thought": "Describe your reasoning for decomposing tasks, interpreting control flows, choosing agent strategies, and sequencing stages.",
    "merge_filtered_workflow": {{
        "Control Flow 0": {{
            "flow_type": "start sequential",
            "flow_desc": "..."
        }},
        "Stage 0": {{
            "stage_id": "stage_0",
            "objective": "...",
            "agent_collaboration": [CoT | SC_CoT | Debate | Reflexion | AnswerGenerate | SpecificFormat | Aggregate | Programmer | Review | Revise"],
            "dependencies": []
        }},
        "Stage 1": {{
            "stage_id": "stage_1",
            "objective": "...",
            "agent_collaboration": [CoT | SC_CoT | Debate | Reflexion | AnswerGenerate | SpecificFormat | Aggregate | Programmer | Review | Revise"],
            "dependencies": [stage_0, ....]
        }},
        .....
        "Control Flow n": {{
            "flow_type": "end sequential",
            "flow_desc": "..."
        }}
    }}
}}
    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_merge_workflow},
    ]

    merge_filtered_workflow ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'merge_filtered_workflow'], 0.0)
    print("================ Merged Filtered Workflow ================\n", merge_filtered_workflow['merge_filtered_workflow'])

    return merge_filtered_workflow['merge_filtered_workflow']
 