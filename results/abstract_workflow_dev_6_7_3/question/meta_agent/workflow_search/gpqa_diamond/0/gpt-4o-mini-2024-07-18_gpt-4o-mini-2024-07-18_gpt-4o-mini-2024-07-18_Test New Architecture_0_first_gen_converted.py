async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Analyze the lifetimes of the two quantum states and the energy difference options, determining the minimum ΔE needed for clear resolution."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing lifetimes and energy differences, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    for lifetime in [10**-9, 10**-8]:
        delta_e_instruction = f"Sub-task 2: Generate initial candidate resolutions for lifetime {lifetime} by computing the minimum ΔE needed for clear resolution."
        delta_e_agent_desc = {
            'instruction': delta_e_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results2 = await self.answer_generate(
            subtask_id="subtask_2",
            cot_agent_desc=delta_e_agent_desc
        )
        agents.append(f"AnswerGenerate agent {results2['cot_agent'].id}, generating candidate resolutions, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
    aggregate_instruction = "Sub-task 3: Aggregate the candidate resolutions and select the correct energy difference option that meets the resolvability criterion."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating candidates, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs