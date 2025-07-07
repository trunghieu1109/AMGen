async def forward_5(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction1 = (
        "Subtask 1: Identify and verify the historical identity of the Rome Protocols and the three Prime Ministers who signed them. "
        "Explicitly cross-check the term 'Rome Protocols' with reliable historical sources and confirm the correct signatories before answering. "
        "Provide detailed reasoning and correct historical context based on taskInfo."
    )
    critic_instruction1 = (
        "Please review the identification and verification of the Rome Protocols and their signatories. "
        "Point out any factual inaccuracies or assumptions and provide corrections if needed."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, verifying Rome Protocols and signatories, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correction: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = (
        "Subtask 2: Based on the verified signatories from Subtask 1, determine which of the three Prime Ministers was assassinated. "
        "Include step-by-step reasoning and validate the accuracy of the signatories before concluding."
    )
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, determining assassinated Prime Minister, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_instruction3 = (
        "Subtask 3: Identify and verify the historical event or context related to the assassination of the Prime Minister determined in Subtask 2. "
        "Ensure the event is correctly linked to the Rome Protocols identified in Subtask 1, with detailed reasoning and fact-checking."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, identifying assassination event context, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_reflect_instruction4 = (
        "Subtask 4: Cross-validate and consolidate the answers from Subtasks 1, 2, and 3. "
        "Detect and flag any inconsistencies or factual errors across the subtasks. "
        "Provide a coherent and factually accurate consolidated answer about the Rome Protocols, the assassinated Prime Minister, and the related event."
    )
    critic_instruction4 = (
        "Please review the consolidated answer for factual accuracy and consistency. "
        "Provide feedback on any discrepancies or errors and suggest corrections."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['answer'], results2['answer'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "answers of subtasks 1, 2, 3"]
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, consolidating and validating answers, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correction: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining consolidated answer, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_reflect_instruction5 = (
        "Subtask 5: Critically validate the consolidated answer from Subtask 4 for factual accuracy, completeness, and alignment with authoritative historical sources. "
        "Provide detailed justification and references if possible."
    )
    critic_instruction5 = (
        "Please fact-check the consolidated answer and provide feedback on its accuracy and completeness. "
        "Suggest any necessary corrections or improvements."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, validating consolidated answer, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correction: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining validation, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_instruction6 = (
        "Subtask 6: Generate a final, clear, and factually accurate answer synthesizing all validated information about the Rome Protocols, the assassinated Prime Minister, and the related event. "
        "Avoid code generation; focus on a comprehensive textual answer suitable for the user query."
    )
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results4['answer'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "consolidated answer", "validation answer"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, generating final refined answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    cot_reflect_instruction7 = (
        "Subtask 7: Perform a final comprehensive reflective check to ensure the final answer fully addresses the query about the Rome Protocols and the assassination context. "
        "Critically assess the entire workflow output for factual correctness and completeness."
    )
    critic_instruction7 = (
        "Please review the final answer for any remaining factual errors or omissions. "
        "Confirm that the answer is accurate, complete, and coherent with known historical facts."
    )
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of final answer", "final answer"]
    }
    critic_desc7 = {
        'instruction': critic_instruction7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=cot_reflect_desc7,
        critic_desc=critic_desc7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, final reflective check, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correction: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining final check, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
