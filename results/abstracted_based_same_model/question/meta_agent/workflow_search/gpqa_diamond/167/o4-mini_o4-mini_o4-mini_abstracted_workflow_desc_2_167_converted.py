async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Parse the question text to extract: (a) the four numbered issues with their descriptions, and (b) the four answer choices labeled A–D, each mapped to the list of issue numbers it references. Output a structured mapping issues{1–4}->description and choices{A–D}->[issue numbers]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing question, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Classify each extracted issue as 'common' or 'not common' using definitions: 'difficult-to-spot' are silent mismatches (e.g. hidden coordinate shifts), whereas 'easy-to-spot' are parse or format errors that trigger immediate failures."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying issues, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction = "Sub-task 2 Reflection: Review the classification mapping. For any borderline cases, ask 'Could this error ever be detected easily? Why or why not?' and adjust the labels if needed. Output a revised mapping issue->'common'/'not common'."
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    reflect_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc2_reflection = {"subtask_id": "subtask_2_reflection", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Reflexion"}
    thinking_reflect, answer_reflect = await cot_agent_reflect(reflect_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, reflecting on classifications, thinking: {thinking_reflect.content}; answer: {answer_reflect.content}")
    for r in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect, answer_reflect], "Please review the classification mapping and indicate if it is final ('True') or requires changes ('False').", r, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, round {r}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        reflect_inputs.extend([thinking_reflect, answer_reflect, feedback])
        thinking_reflect, answer_reflect = await cot_agent_reflect(reflect_inputs, cot_reflect_instruction, r+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining classifications, thinking: {thinking_reflect.content}; answer: {answer_reflect.content}")
    sub_tasks[-1] = f"Sub-task 2 output: thinking - {thinking_reflect.content}; answer - {answer_reflect.content}"
    subtask_desc2_reflection["response"] = {"thinking": thinking_reflect, "answer": answer_reflect}
    logs.append(subtask_desc2_reflection)
    print("Step 2 Reflection: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3: Verify the Sub-task 2 classifications against a curated knowledge base of known difficult-to-spot errors ['chr-prefix mismatch', 'reference build mismatch', 'silent ID conversion issues']. Then, for each answer choice A–D, check whether all its referenced issues are tagged 'common'. Output a mapping choice->validity flag (true/false)."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "thinking of subtask 2 reflection", "answer of subtask 2 reflection"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking_reflect, answer_reflect], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, verifying choices, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Select the single answer choice with validity flag=true (the most comprehensive set of common difficult-to-spot error sources) and return that letter."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, selecting final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs