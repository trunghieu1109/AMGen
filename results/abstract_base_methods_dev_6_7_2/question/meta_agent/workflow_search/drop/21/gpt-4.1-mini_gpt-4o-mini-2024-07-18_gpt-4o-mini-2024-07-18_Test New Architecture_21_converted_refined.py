async def forward_21(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Sub-task 1: Extract and analyze Peyton Manning's passing statistics from the passage to identify the number of passes thrown (attempts) with context from the task information."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing Peyton Manning's passing stats, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = "Sub-task 2: Decompose and logically sequence the extracted information to confirm the exact number of passes Peyton Manning threw (attempts), ensuring clarity between attempts and completions."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing and sequencing info, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    reflexion_instruction = "Sub-task 3: Review the derived answer to ensure it correctly addresses the question: 'How many passes did Peyton Manning throw?'. Explicitly confirm that 'passes thrown' corresponds to total attempts, not completions, to avoid confusion."
    critic_instruction = "Please review the answer for correctness, focusing on whether the number corresponds to passes thrown (attempts) as asked in the question. Provide feedback and correctness judgment."
    cot_reflect_desc = {
        'instruction': reflexion_instruction,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, reviewing answer correctness, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correct: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    sc_cot_instruction = "Sub-task 4: Aggregate multiple reasoning paths to select the most consistent and semantically correct final answer for the number of passes Peyton Manning threw, explicitly distinguishing attempts from completions."
    sc_cot_desc = {
        'instruction': sc_cot_instruction,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2", "thinking and answer of subtask 3"]
    }
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_desc=sc_cot_desc,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results4['list_thinking']):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, reasoning path {idx+1}, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs