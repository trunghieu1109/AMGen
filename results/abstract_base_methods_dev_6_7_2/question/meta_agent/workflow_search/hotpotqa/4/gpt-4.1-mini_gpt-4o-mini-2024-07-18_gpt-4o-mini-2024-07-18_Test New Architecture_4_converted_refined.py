async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    magazine_names = ["Woman's Era", "Naj"]
    magazine_types = {}
    
    # Stage 1: Verify and identify the type of 'Woman's Era' magazine using Reflexion
    cot_reflect_instruction1 = "Sub-task 1: Identify and explicitly verify the type or category of the magazine 'Woman's Era' by gathering reliable information or stating if data is unavailable. Provide a concise final answer without reasoning."
    critic_instruction1 = "Please review the identification and verification of 'Woman's Era' magazine type and provide feedback on its accuracy and limitations."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo, "Woman's Era"],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "magazine name: Woman's Era"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, verifying 'Woman's Era' magazine type, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback on 'Woman's Era' verification, thinking: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer for 'Woman's Era', thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    magazine_types["Woman's Era"] = results1['answer'].content
    
    # Stage 2: Verify and identify the type of 'Naj' magazine using Reflexion
    cot_reflect_instruction2 = "Sub-task 2: Identify and explicitly verify the type or category of the magazine 'Naj' by gathering reliable information or stating if data is unavailable. Avoid assuming similarity to other magazines. Provide a concise final answer without reasoning."
    critic_instruction2 = "Please review the identification and verification of 'Naj' magazine type and provide feedback on its accuracy and limitations."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, "Naj"],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "magazine name: Naj"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, verifying 'Naj' magazine type, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback on 'Naj' verification, thinking: {results2['list_feedback'][i].content}; correct: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer for 'Naj', thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    magazine_types["Naj"] = results2['answer'].content
    
    # Stage 3: Integrate and synthesize magazine types using Self-Consistency Chain-of-Thought
    cot_sc_instruction3 = "Sub-task 3: Based on the verified types of 'Woman's Era' and 'Naj', evaluate their consistency and synthesize a coherent, concise final answer describing what kind of magazines they are. Consider alternative categorizations and confidence levels."
    N = self.max_sc
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, magazine_types["Woman's Era"], magazine_types["Naj"]],
        'temperature': 0.5,
        'context': ["user query", "verified type of Woman's Era", "verified type of Naj"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, synthesizing magazine types, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 4: Review and fact-check the integrated answer using Reflexion
    cot_reflect_instruction4 = "Sub-task 4: Critically review and fact-check the integrated magazine type answer from Sub-task 3. Correct any inaccuracies or explicitly state uncertainty if information is insufficient. Provide a concise, final verified answer."
    critic_instruction4 = "Please review the integrated magazine type answer for accuracy, completeness, and validity, and provide corrections or confirmations."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reviewing integrated answer, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback on integrated answer, thinking: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
