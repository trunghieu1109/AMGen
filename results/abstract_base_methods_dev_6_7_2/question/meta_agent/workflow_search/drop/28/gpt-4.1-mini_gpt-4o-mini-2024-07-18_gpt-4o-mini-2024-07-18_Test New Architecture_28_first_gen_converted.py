async def forward_28(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    for iteration in range(1):
        # Stage 1: Iterative Quality Enhancement (subtask_1)
        cot_reflect_instruction = "Subtask 1: Iteratively evaluate and refine the understanding of the passage to improve clarity, consistency, and completeness of the inferred cause of Mirwais' death."
        critic_instruction = "Please review the refined understanding and provide feedback on its clarity, consistency, and completeness."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results1 = await self.reflexion(
            subtask_id="subtask_1",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining understanding of Mirwais' cause of death, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results1['list_feedback']))):
            agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
            if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
                agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining final answer, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
        sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
        logs.append(results1['subtask_desc'])

        # Stage 0: Construct Logical Reasoning Sequence (subtask_2) depends on subtask_1
        cot_instruction2 = "Subtask 2: Decompose the passage information into a logical sequence of reasoning steps to derive the cause of Mirwais' death and generate the answer, using the refined understanding from Subtask 1."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.0,
            'context': ["user query", "refined thinking of subtask 1", "refined answer of subtask 1"]
        }
        results2 = await self.answer_generate(
            subtask_id="subtask_2",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing passage to logical reasoning steps, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])

    # Control Flow 2: end_loop
    # Control Flow 3: end_sequential

    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'], sub_tasks, agents)
    return final_answer, logs