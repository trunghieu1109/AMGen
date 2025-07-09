async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Define spontaneous symmetry breaking and the Nambu–Goldstone theorem in both particle physics and condensed-matter contexts, outlining the necessary conditions for Nambu–Goldstone bosons to appear."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, define SSB and NG theorem, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot2_instruction = "Sub-task 2: List the specific continuous symmetry that is spontaneously broken for each candidate—magnon, skyrmion, pion, and phonon—including a reminder that breaking continuous translational symmetry in a crystal produces acoustic phonons as Goldstone modes."
    cot2_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot2_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot2_agent([taskInfo, thinking1, answer1], cot2_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot2_agent.id}, list broken symmetries, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot3_instruction = "Sub-task 3: For each particle, independently evaluate whether it meets the criteria of a Nambu–Goldstone boson based on the symmetries identified in Sub-task 2."
    N = self.max_sc
    cot3_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot3_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for agent in cot3_agents:
        thinking3i, answer3i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, evaluate NG status, thinking: {thinking3i.content}; answer: {answer3i.content}")
        possible_answers.append(answer3i.content)
        thinking_map[answer3i.content] = thinking3i
        answer_map[answer3i.content] = answer3i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3, answer3 = thinking_map[answer3_content], answer_map[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction = "Sub-task 4: Conduct a focused debate on whether acoustic phonons are genuine Nambu–Goldstone modes from broken translational symmetry or not."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    Nround = self.max_round
    all_thinking4 = [[] for _ in range(Nround)]
    all_answer4 = [[] for _ in range(Nround)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(Nround):
        for agent in debate_agents:
            if r == 0:
                t4, a4 = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                t4, a4 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t4.content}; answer: {a4.content}")
            all_thinking4[r].append(t4)
            all_answer4[r].append(a4)
    final_decision4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Adjudicate the debate and provide a final stance on phonon classification.", is_sub_task=True)
    agents.append(f"Final Decision agent, adjudicating phonon classification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 5: Reflect on the preliminary classifications of all four particles by comparing them against the formal definition of Nambu–Goldstone bosons and known textbook examples to ensure no misclassification persists."
    cot_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Reflexion"}
    inputs5 = [taskInfo, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = await cot_reflect(inputs5, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect.id}, initial reflection, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], "Please review the preliminary classifications and provide any limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        inputs5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_reflect(inputs5, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect.id}, refined reflection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot6_instruction = "Sub-task 6: Compile a final mapping of each particle to a Boolean flag indicating whether it is associated with a spontaneously broken symmetry (true = NG boson, false = not NG boson)."
    cot6_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot6_instruction, "context": ["user query", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot6_agent([taskInfo, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot6_agent.id}, compile final mapping, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot7_instruction = "Sub-task 7: Identify which particle in the mapping from Sub-task 6 has a false flag (i.e., is not associated with spontaneous symmetry breaking)."
    cot7_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot7_instruction, "context": ["user query", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot7_agent([taskInfo, thinking6, answer6], cot7_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot7_agent.id}, identify false flag particle, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot8_instruction = "Sub-task 8: Convert the identified particle from Sub-task 7 into the corresponding answer choice letter (A, B, C, or D) where Magnon (A), Skyrmion (B), Pion (C), Phonon (D)."
    cot8_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot8_instruction, "context": ["user query", "thinking of subtask 7", "answer of subtask 7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot8_agent([taskInfo, thinking7, answer7], cot8_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot8_agent.id}, convert to letter, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs