async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Extract and summarize given reaction steps (SC-CoT)
    cot_sc_instruction_1 = "Sub-task 1: Extract and summarize all given reaction steps, reagents, and transformation types from the query."
    N1 = self.max_sc
    sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction_1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_1[i].id}, summarizing reaction steps, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings_1.append(thinking1_i)
        possible_answers_1.append(answer1_i)
    final_sc_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Sub-task 1: Given all the above thinking and answers, find the most consistent and correct summary of the reaction steps, reagents, and transformations from the query."
    thinking1, answer1 = await final_sc_1([taskInfo] + possible_thinkings_1 + possible_answers_1, final_instr_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Deduce structure of product 1 (SC-CoT)
    cot_sc_instruction_2 = "Sub-task 2: Deduce the structure of product 1 by applying NaH deprotonation and benzylation to the starting cyclohexanone derivative."
    N2 = self.max_sc
    sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings_2 = []
    possible_answers_2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction_2, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_2[i].id}, deducing product 1, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings_2.append(thinking2_i)
        possible_answers_2.append(answer2_i)
    final_sc_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Given all the above thinking and answers, find the most consistent and correct structure for product 1."
    thinking2, answer2 = await final_sc_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Elucidate structures of products 2 and 3 (Debate)
    debate_instr_3 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3 = "Sub-task 3: Elucidate the structures of product 2 (tosyl hydrazone) and product 3 (Shapiro-type rearrangement product) based on hydrazone formation and n-BuLi treatment." + debate_instr_3
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3 = [[] for _ in range(self.max_round)]
    all_answer_3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction_3, "context": ["user query", "thinking2", "answer2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                inp = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3_i, answer3_i = await agent(inp, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking_3[r].append(thinking3_i)
            all_answer_3[r].append(answer3_i)
    final_decision_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_3([taskInfo, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1], final_instr_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine structure of product 4 after hydrogenolysis (SC-CoT)
    cot_sc_instruction_4 = "Sub-task 4: Determine the structure of product 4 formed after Pd/C hydrogenolysis, including removal of protecting groups."
    N4 = self.max_sc
    sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings_4 = []
    possible_answers_4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction_4, "context": ["user query", "thinking3", "answer3"], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await sc_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_4[i].id}, determining product 4, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings_4.append(thinking4_i)
        possible_answers_4.append(answer4_i)
    final_sc_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Given all the above thinking and answers, find the most consistent and correct structure for product 4."
    thinking4, answer4 = await final_sc_4([taskInfo, thinking3, answer3] + possible_thinkings_4 + possible_answers_4, final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare derived structure to options and select (Debate)
    debate_instr_5 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = "Sub-task 5: Compare the derived structure of product 4 to the four multiple-choice options and select the one that matches exactly." + debate_instr_5
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_5 = [[] for _ in range(self.max_round)]
    all_answer_5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction_5, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                inp5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5_i, answer5_i = await agent(inp5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking_5[r].append(thinking5_i)
            all_answer_5[r].append(answer5_i)
    final_decision_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_5([taskInfo, thinking4, answer4] + all_thinking_5[-1] + all_answer_5[-1], final_instr_5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs