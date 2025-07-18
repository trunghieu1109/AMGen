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

async def specific_task_decomposition(meta_model, query, collaboration_dependencies, aw_flow = None, prev_task_decomposition = None, evaluation = None, is_refinement = False):
    # processed_aw_flow = [v for k, v in aw_flow.items()]
    
    if not is_refinement:
        user_prompt = f"""
You are an agent specializing in task decomposition. Given a query, your job is to break it into subtasks with clear objectives and dependencies. 

[Inputs]
- Query: {query}
- Abstract Workflow: {aw_flow}
- Dependencies and Agent Collaboration: {collaboration_dependencies}

[Instructions]
1. Do not decompose query to too small / trivial subtasks. Maximum 5 subtasks.
2. For each subtask, include:
- Objective: Objective that need to address in this subtask.
- Dependencies: Subtasks that must be completed before this one.
- Agent Collaboration Patterns: SC_CoT | Debate | Reflexion | CoT.  Thứ tự ưu tiên khi lựa chọn các patterns với khả năng reasoning mạnh, ví dụ như Debate > SC_CoT > Reflexion > CoT. 
3. A subtask can draw context from multiple previous steps to provide additional information for the current step, not just from the immediately preceding one. Therefore, try to choose appropriately. The dependencies are a useful reference: if there is a dependency from stage A to stage B, it means that steps in A can serve as context for steps in B.
5. If the output of a certain step can support or is related to a subtask, create a dependency between those subtasks
6. Outputs of one subtask should serve as proper inputs to the next.
7. Adhere to the structure defined by the Abstract Workflow to decompose the query into essential steps for solving the problem. Ensure strict compliance with the workflow and concretize each step using details extracted from the query.
8. Within each Stage, there may be multiple subtasks.
9. Dependencies should be established based on the Abstract Workflow structure as well as the intrinsic relationship between the query content and individual subtasks. For instance, if a subtask produces information A that is relevant or supportive to subsequent subtasks, a dependency should be explicitly defined between them.

[Output format — JSON]
{{
    "thought": "<your brief reasoning on how you break the query>",
    "task_decomposition": {{
        "stage_1": {{
            "subtask_1": {{
                "objective": "...",
                "dependencies": [],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }},
            "subtask_2": {{
                "objective": "...",
                "dependencies": ["subtask_1", ...],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }}
        }},
        "stage_2": {{
        ...
        }}
    }}
}}
        """
        
        # print("\n==================== planning prompt =================\n", user_prompt)
    
    else:
        failure_reason = [eva.get("failure_reason", "") for eva in evaluation]
        user_prompt = f"""
You are an agent specializing in task decomposition. The previous task decomposition encountered several issues that led to inefficiency. Therefore, please revise the task decomposition based on the evaluation results from earlier.

[Inputs]
- Query: {query}
- Evaluation from previous attempts: {evaluation}
- Previous decomposition: {prev_task_decomposition}
- Dependencies and Agent Collaboration: {collaboration_dependencies}

[Instructions]
1. Do not decompose query to too small / trivial subtasks.
2. For each subtask, include:
- Objective: Objective that need to address in this subtask. Leverage feedback from previous attempts. Especially, embed the failure reason / feedback to enhance the objective of each subtask.
- Dependencies: Subtasks that must be completed before this one.
- Agent Collaboration Patterns: SC_CoT | Debate | Reflexion | CoT. Thứ tự ưu tiên khi lựa chọn các patterns với khả năng reasoning mạnh, ví dụ như Debate > SC_CoT > Reflexion > CoT. 
3. Use feedback (if any) to:
- Add những feedback về lỗi trong qua trình reasoning vào trong objective của subtasks để tránh lặp lại lỗi sai. Nguyên nhân bị sai: {failure_reason}
- Xem xét các gợi ý và feedback và sửa lại task decomposition:
    + Fix missing dependencies between subtasks.
    + Decompose subtasks to smaller subtasks.
    + Clarify unclear steps.
    + Refine objective of each subtasks: Revise them thoroughly to avoid repeating the mistakes highlighted in the experts' feedback.
4. subtask can draw context from multiple previous steps to provide additional information for the current step, not just from the immediately preceding one. Therefore, try to choose appropriately. The dependencies are a useful reference: if there is a dependency from stage A to stage B, it means that steps in A can serve as context for steps in B.
6. If the output of a certain step can support or is related to a subtask, create a dependency between those subtasks
7. Ensure better granularity, clarity, and logical flow than past attempts.
8. Ensure that all suggestions and feedback are integrated into the task decomposition by explicitly embedding them within the instructions of the corresponding subtasks.
9. Dependencies should be established based on the Abstract Workflow structure as well as the intrinsic relationship between the query content and individual subtasks. For instance, if a subtask produces information A that is relevant or supportive to subsequent subtasks, a dependency should be explicitly defined between them.

[Output format — JSON]
{{
    "thought": "<your brief reasoning on how you refined the breakdown>",
    "task_decomposition": {{
        "stage_1": {{
            "subtask_1": {{
                "objective": "...",
                "dependencies": [],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }},
            "subtask_2": {{
                "objective": "...",
                "dependencies": ["subtask_1", ....],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }}
        }},
        "stage_2": {{
        ...
        }}
    }}
}}
        """

    msg_list = [
        {"role": "user", "content": user_prompt},
    ]

    task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'task_decomposition'], 0.0)
            
    return task_decomposition['task_decomposition']
    
async def generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, query):
    user_prompt_generate_workflow = f"""
You are tasked with generating a concrete multi-stage agentic workflow using techniques like Chain-of-Thought, Self-Consistency, Reflexion, or Debate. Your goal is to implement the workflow for the provided query, following the task decomposition and improving from previous feedback.

Ensure this code is runnable. Double-check many times.

[Inputs]
1. Query: {query}
2. Task Decomposition: {task_decomposition}
3. Abstract Workflow Template: {ABSTRACTED_WORKFLOW_TEMPLATE}. Return code in format of a function `forward`
4. Interaction Pattern Guide: {interaction_pattern}

[Requirements]
1. Format your output in JSON format as:
{{
  "thought": "Brief reasoning behind your workflow implementation.",
  "code": "Python code implementing the concrete agentic workflow."
}}
2. For each subtask:
   - Ensure that each subtask strictly follows the instruction defined in the task decomposition and incorporates relevant evaluation feedback and suggestions.
   - Apply the exactly agent collaboration patterns to each subtask as specified in task decomposition.
   - Pass the outputs from Subtask A to Subtask B in accordance with the defined dependencies between the subtasks.
   - Follow the task decomposition strictly (stage, order, dependencies).
   - Implement using appropriate agent collaboration pattern (CoT, SC-CoT, Reflexion, Debate).
   - Replace any placeholder with task-specific content grounded in the query.

3. Preserve logic from abstract workflow:
   - Use same number of stages.
   - Link subtasks clearly via outputs/inputs.
   - Synthesize final answer using `make_final_answer`.
   
5. Use variables: `self.node_model`, `self.debate_role`, `self.max_sc`, `self.max_round`.
6. Do not use LaTeX.
7. Do not include comments in code.

[Goal]
Ensure all subtasks are well-structured, grounded, and correctly executed. Output a fully working Python-based workflow.
    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_generate_workflow},
    ]

    next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'code'], 0.0)
    print("================ Generated Multi-agent system ================\n", next_solution['code'])
    next_solution['code'] = next_solution['code'].replace("{{", "{")
    next_solution['code'] = next_solution['code'].replace("}}", "}")
    
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

async def merge_filtered_workflow(meta_model, filtered_sequential_only_workflow, filtered_loop_contain_workflow, filtered_conditional_contain_workflow):
    user_prompt_merge_workflow = f"""
You are a specialized AI agent for analyzing and merging task workflows represented in structured JSON format. Your task is to intelligently merge two similar but not identical workflows into a unified, optimized version that preserves both shared and unique subtasks, while maintaining logical structure and relative control flow positions.

### 🔧 INPUT

You will be given **two workflows**, each represented as a list of subtasks in JSON format. Each subtask has the following fields:
- `subtask_id`: unique identifier
- `subtask_name`: human-readable name
- `abstracted_objective`: a concise description of what the subtask accomplishes
- `agent_collaboration`: either a string (e.g., "logic code") or a list of collaboration methods (e.g., `["CoT", "AnswerGenerate"]`)
- `dependencies` (optional): list of subtask_ids this subtask depends on

Each workflow may also contain **control flow markers**:
- `start_sequential` / `end_sequential`
- `start_loop` / `end_loop`
- `start_conditional` / `end_conditional`
- `start_true_branch` / `end_true_branch`
- `start_false_branch` / `end_false_branchs`

### 📌 TASK

1. **Merge the two workflows** into a single workflow list that:
   - Keeps the relative position of control flows (`start_sequential`, `start_loop`, etc.) **exactly as they appear** in each respective flow.
   - If two subtasks serve **identical purposes** and appear at similar relative positions, **merge them into one**.
     - Consider subtasks as equivalent if their `abstracted_objective` and `agent_collaboration` are very similar and they appear in the same structural place (e.g., inside a loop).
   - Keep **unique subtasks** from each workflow, ensuring there is **no redundant repetition**.
   - Adjust `dependencies` accordingly in the merged version.
   - If a subtask is present in only one workflow but logically fits into the merged structure, include it and annotate it with `"optional": true`.

2. **Ensure logical consistency**: 
   - Do not place subtasks outside of their valid control flow (e.g., don't put a looping subtask outside a loop).
   - Do not duplicate identical or redundant subtasks.
   - Maintain correct ordering of dependencies.

3. **Avoid**:
   - Breaking control flow structure (e.g., nesting start/end incorrectly).
   - Losing important agent collaboration roles or objectives.
   - Blindly appending all subtasks together.

### 🧾 OUTPUT FORMAT

Return results in JSON format, which include `thought` (Describe your reasoning for decomposing tasks, interpreting control flows, choosing agent strategies, and sequencing subtasks.)
and `merge_filtered_workflow` that represents the merged workflow.
In merge_filtered_workflow, it could contain `Control Flow` (start_sequential, end_sequential, ....) or `Stage`. Each stage include multiple subtasks. Each of them contains:
- `subtask_id`
- `subtask_name`
- `abstracted_objective`
- `agent_collaboration`
- `dependencies` (optional)

You will be provided with:
- Filtered Sequential Only Workflow:
{filtered_sequential_only_workflow}

- Filtered Loop Contain Workflow:
{filtered_loop_contain_workflow}

- Filtered Conditional Contain Workflow:
{filtered_conditional_contain_workflow}

### 🧠 Reasoning Strategy

While merging, follow these reasoning steps:
- Align control flow structure first.
- Identify matching subtasks by comparing their names, objectives, collaboration patterns, and structural position.
- Merge identical subtasks, keep unique ones if relevant.

Example:
{{
    "thought": "Describe your reasoning for decomposing tasks, interpreting control flows, choosing agent strategies, and sequencing subtasks.",
    "merge_filtered_workflow": {{
        "Control Flow 0": {{
            "flow_type": "start sequential",
            "flow_desc": "..."
        }},
        "Stage 0": {{
            "subtask_1": {{
                "objective": "...",
                "agent_collaboration": "CoT | SC_CoT | Debate | Reflexion | AnswerGenerate | SpecificFormat | Aggregate | CodeGenerator | Programmer | Review | Revise",
                "dependencies": []
            }},
            "subtask_2": {{
                "objective": "...",
                "agent_collaboration": "...",
                "dependencies": ["subtask_1"]
            }}
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
 