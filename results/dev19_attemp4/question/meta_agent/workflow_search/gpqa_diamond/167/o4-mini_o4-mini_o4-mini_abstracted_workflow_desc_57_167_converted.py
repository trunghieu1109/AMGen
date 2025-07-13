async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1 - Subtask 1: Extract and define issues
    cot_instruction = (
        "Sub-task 1: Extract and clearly define each of the four issues in the question (A: mutually incompatible data formats; "
        "B: the ‘chr’/‘no chr’ confusion; C: reference assembly mismatch; D: incorrect ID conversion) and define what 'difficult-to-spot' errors are."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 - Subtask 2: Score silent-failure likelihood with examples (SC-CoT)
    cot_sc_instruction = (
        "Sub-task 2: Assign a silent-failure likelihood score (1-5) for each issue A-D, with at least one concrete real-world example per issue, distinguishing silent errors from immediate failures."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction,
                     "context": ["user query", thinking1, answer1], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and evidence-backed likelihood scores and rationales per issue."
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
        "Sub-task 2: Synthesize and choose the most consistent scores and rationales." + final_instr_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2 - Subtask 3: Debate to reconcile disagreements
    debate_instr3 = (
        "Sub-task 3: Reconcile disagreements in the scoring or examples from Sub-task 2, with agents debating conflicting views and applying a tie-breaking rule to output a single agreed-upon score and rationale for each issue. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    N_max3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max3)]
    all_answer3 = [[] for _ in range(N_max3)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instr3,
                     "context": ["user query", thinking2, answer2], "agent_collaboration": "Debate"}
    for r in range(N_max3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instr3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs, debate_instr3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and reconcile a final set of scores and rationales per issue."
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Reconcile final consensus." + final_instr_3,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3 - Subtask 4: Identify most common hidden error sources (SC-CoT)
    cot_sc_instruction4 = (
        "Sub-task 4: Based on the agreed scores and rationales from Sub-task 3, identify which issues most frequently cause silent, hard-to-spot errors and determine the best combination of issues (pair, triple, or all four)."
    )
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4,
                     "context": ["user query", thinking3, answer3], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, choose the combination of issues that best matches the question’s focus on the most common hidden error sources."
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize final combination." + final_instr_4,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 4 - Subtask 5: Select answer choice (Debate)
    debate_instruction5 = (
        "Sub-task 5: Select the answer choice (choice1–choice4) that exactly corresponds to the combination identified in Sub-task 4. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5,
                     "context": ["user query", thinking4, answer4], "agent_collaboration": "Debate"}
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Final selection of answer choice." + final_instr_5,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs