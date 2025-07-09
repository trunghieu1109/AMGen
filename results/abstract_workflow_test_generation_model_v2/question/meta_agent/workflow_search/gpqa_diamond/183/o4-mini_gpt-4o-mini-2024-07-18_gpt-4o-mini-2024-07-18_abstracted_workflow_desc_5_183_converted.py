async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 0_1: Identify and catalog all functional groups (tert-butyl, ethoxy, nitro) in 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, including their exact ring positions and electronic activating/deactivating and directing characteristics."
    cot_agent0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0_1 = {"subtask_id": "subtask_0_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking0_1, answer0_1 = await cot_agent0_1([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent0_1.id}, identifying functional groups, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc0_1['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc0_1)
    print("Step 1: ", sub_tasks[-1])
    debate_instruction0_2 = "Sub-task 0_2: Characterize the starting material benzene: confirm absence of substituents and establish its baseline reactivity and directing behavior."
    debate_agents0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking0_2 = [[] for _ in range(self.max_round)]
    all_answer0_2 = [[] for _ in range(self.max_round)]
    subtask_desc0_2 = {"subtask_id": "subtask_0_2", "instruction": debate_instruction0_2, "context": ["user query"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents0_2:
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instruction0_2, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo] + all_thinking0_2[r-1] + all_answer0_2[r-1], debate_instruction0_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, characterizing benzene, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking0_2[r].append(thinking)
            all_answer0_2[r].append(answer)
    final_decision_agent0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_2, answer0_2 = await final_decision_agent0_2([taskInfo] + all_thinking0_2[-1] + all_answer0_2[-1], "Sub-task 0_2: Make final characterization decision of benzene.", is_sub_task=True)
    agents.append(f"Final Decision agent, benzene characterization, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    subtask_desc0_2['response'] = {"thinking": thinking0_2, "answer": answer0_2}
    logs.append(subtask_desc0_2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction1_1 = "Sub-task 1_1: For every reagent/condition in choices A–D, classify the reaction type and specify the net functional-group change on a generic benzene ring."
    cot_agents1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers1_1 = []
    thinking_map1_1 = {}
    answer_map1_1 = {}
    subtask_desc1_1 = {"subtask_id": "subtask_1_1", "instruction": cot_sc_instruction1_1, "context": ["user query", "thinking0_1", "answer0_1", "thinking0_2", "answer0_2"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents1_1[i]([taskInfo, thinking0_1, answer0_1, thinking0_2, answer0_2], cot_sc_instruction1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1_1[i].id}, classifying reagents, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers1_1.append(answer.content)
        thinking_map1_1[answer.content] = thinking
        answer_map1_1[answer.content] = answer
    answer1_1_content = Counter(possible_answers1_1).most_common(1)[0][0]
    thinking1_1 = thinking_map1_1[answer1_1_content]
    answer1_1 = answer_map1_1[answer1_1_content]
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1_1)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction1_2 = "Sub-task 1_2: Summarize directing effects and activating/deactivating influence for each reaction type."
    cot_agent1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1_2 = {"subtask_id": "subtask_1_2", "instruction": cot_instruction1_2, "context": ["user query", "thinking1_1", "answer1_1"], "agent_collaboration": "CoT"}
    thinking1_2, answer1_2 = await cot_agent1_2([taskInfo, thinking1_1, answer1_1], cot_instruction1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1_2.id}, summarizing directing effects, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc1_2)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction2_1a = "Sub-task 2_1a: Map each of the four choice sequences (A–D) step-by-step: name each intermediate, label new functional group and position, and show how it serves as substrate for next step."
    cot_agent2_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1a = {"subtask_id": "subtask_2_1a", "instruction": cot_instruction2_1a, "context": ["user query", "thinking1_2", "answer1_2"], "agent_collaboration": "CoT"}
    thinking2_1a, answer2_1a = await cot_agent2_1a([taskInfo, thinking1_2, answer1_2], cot_instruction2_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2_1a.id}, mapping intermediates, thinking: {thinking2_1a.content}; answer: {answer2_1a.content}")
    sub_tasks.append(f"Sub-task 2_1a output: thinking - {thinking2_1a.content}; answer - {answer2_1a.content}")
    subtask_desc2_1a['response'] = {"thinking": thinking2_1a, "answer": answer2_1a}
    logs.append(subtask_desc2_1a)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction2_1b = "Sub-task 2_1b: Mechanistic Check: verify each intermediate mapping for chemical feasibility, confirm required functional groups and no directing-effect violations."
    cot_agent2_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2_1b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1b = {"subtask_id": "subtask_2_1b", "instruction": cot_reflect_instruction2_1b, "context": ["user query", "thinking2_1a", "answer2_1a"], "agent_collaboration": "Reflexion"}
    cot_inputs2_1b = [taskInfo, thinking2_1a, answer2_1a]
    thinking2_1b, answer2_1b = await cot_agent2_1b(cot_inputs2_1b, cot_reflect_instruction2_1b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2_1b.id}, initial mechanistic check, thinking: {thinking2_1b.content}; answer: {answer2_1b.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent2_1b([taskInfo, thinking2_1b, answer2_1b], "Review the mechanistic check and flag any issues.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2_1b.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs2_1b += [thinking2_1b, answer2_1b, feedback]
        thinking2_1b, answer2_1b = await cot_agent2_1b(cot_inputs2_1b, cot_reflect_instruction2_1b, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2_1b.id}, refined mechanistic check, thinking: {thinking2_1b.content}; answer: {answer2_1b.content}")
    sub_tasks.append(f"Sub-task 2_1b output: thinking - {thinking2_1b.content}; answer - {answer2_1b.content}")
    subtask_desc2_1b['response'] = {"thinking": thinking2_1b, "answer": answer2_1b}
    logs.append(subtask_desc2_1b)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction2_2 = "Sub-task 2_2: Compile refined reaction-pathway summary for each choice A–D, listing reagents, conditions, intermediates, and mechanistic annotations."
    debate_agents2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2_2 = [[] for _ in range(self.max_round)]
    all_answer2_2 = [[] for _ in range(self.max_round)]
    subtask_desc2_2 = {"subtask_id": "subtask_2_2", "instruction": debate_instruction2_2, "context": ["user query", "thinking2_1b", "answer2_1b"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents2_2:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2_1b, answer2_1b], debate_instruction2_2, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2_1b, answer2_1b] + all_thinking2_2[r-1] + all_answer2_2[r-1], debate_instruction2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compiling pathway summary, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking2_2[r].append(thinking)
            all_answer2_2[r].append(answer)
    final_decision_agent2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent2_2([taskInfo] + all_thinking2_2[-1] + all_answer2_2[-1], "Sub-task 2_2: Finalize refined pathway summaries.", is_sub_task=True)
    agents.append(f"Final Decision agent, pathway summaries, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 7: ", sub_tasks[-1])
    debate_instruction3_1 = "Sub-task 3_1: Evaluate each refined pathway for overall feasibility and predicted yield, checking directing conflicts, deactivation issues, and mechanistic notes."
    debate_agents3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3_1 = [[] for _ in range(self.max_round)]
    all_answer3_1 = [[] for _ in range(self.max_round)]
    subtask_desc3_1 = {"subtask_id": "subtask_3_1", "instruction": debate_instruction3_1, "context": ["user query", "thinking2_2", "answer2_2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents3_1:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2_2, answer2_2], debate_instruction3_1, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2_2, answer2_2] + all_thinking3_1[r-1] + all_answer3_1[r-1], debate_instruction3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating feasibility, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3_1[r].append(thinking)
            all_answer3_1[r].append(answer)
    final_decision_agent3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent3_1([taskInfo] + all_thinking3_1[-1] + all_answer3_1[-1], "Sub-task 3_1: Final decision on pathway feasibility.", is_sub_task=True)
    agents.append(f"Final Decision agent, feasibility evaluation, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 8: ", sub_tasks[-1])
    cot_instruction3_2 = "Sub-task 3_2: Rank the four sequences by synthetic efficiency, mechanistic soundness, and expected yield; select the best choice (A, B, C, or D)."
    cot_agent3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_2 = {"subtask_id": "subtask_3_2", "instruction": cot_instruction3_2, "context": ["user query", "thinking3_1", "answer3_1"], "agent_collaboration": "CoT"}
    thinking3_2, answer3_2 = await cot_agent3_2([taskInfo, thinking3_1, answer3_1], cot_instruction3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3_2.id}, ranking sequences, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3_2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 9: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs