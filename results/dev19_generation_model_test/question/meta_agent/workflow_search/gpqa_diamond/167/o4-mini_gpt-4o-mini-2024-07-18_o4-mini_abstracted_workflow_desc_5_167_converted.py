async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0, Sub-task 1: Extract sources and choices (SC_CoT)
    cot_sc_instruction0 = "Sub-task 0.1: Extract and list the four potential error sources and the four answer choices from the query."
    N0 = self.max_sc
    sc_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id": "stage0_subtask1", "instruction": cot_sc_instruction0, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N0):
        thinking0_i, answer0_i = await sc_agents0[i]([taskInfo], cot_sc_instruction0, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents0[i].id}, extracting sources and choices, thinking: {thinking0_i.content}; answer: {answer0_i.content}")
        possible_thinkings0.append(thinking0_i)
        possible_answers0.append(answer0_i)
    final_instr0 = "Given all the above thinking and answers, find the most consistent and correct extraction of error sources and answer choices."
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent0(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        "Sub-task 0.1: Synthesize and choose the most consistent extraction." + final_instr0,
        is_sub_task=True)
    sub_tasks.append(f"Stage0_Subtask1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])
    # Stage 1, Sub-task 1: Classify error sources (SC_CoT)
    cot_sc_instruction1 = "Sub-task 1.1: Analyze each extracted error source and classify it by its underlying issue type (file format incompatibility, coordinate system mismatch, identifier mapping error)."
    N1 = self.max_sc
    sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "stage1_subtask1", "instruction": cot_sc_instruction1, "context": ["user query", "extraction thinking", "extraction answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents1[i]([taskInfo, thinking0, answer0], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents1[i].id}, classifying sources, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_instr1 = "Given all the above thinking and answers, find the most consistent and correct classification of error sources."
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_class, answer1_class = await final_decision_agent1(
        [taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1,
        "Sub-task 1.1: Synthesize and choose the most consistent classifications." + final_instr1,
        is_sub_task=True)
    sub_tasks.append(f"Stage1_Subtask1 output: thinking - {thinking1_class.content}; answer - {answer1_class.content}")
    subtask_desc1['response'] = {"thinking": thinking1_class, "answer": answer1_class}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2, Sub-task 1: Generate examples (Debate)
    debate_instr2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction2 = "Sub-task 2.1: Generate concrete example scenarios or pipeline contexts in which each classified error leads to difficult-to-spot incorrect results." + debate_instr2
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R2 = self.max_round
    all_thinking2 = [[] for _ in range(R2)]
    all_answer2 = [[] for _ in range(R2)]
    subtask_desc2 = {"subtask_id": "stage2_subtask1", "instruction": debate_instruction2, "context": ["user query", "classification thinking", "classification answer"], "agent_collaboration": "Debate"}
    for r in range(R2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                t2, a2 = await agent([taskInfo, thinking1_class, answer1_class], debate_instruction2, r, is_sub_task=True)
            else:
                inputs2 = [taskInfo, thinking1_class, answer1_class] + all_thinking2[r-1] + all_answer2[r-1]
                t2, a2 = await agent(inputs2, debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t2.content}; answer: {a2.content}")
            all_thinking2[r].append(t2)
            all_answer2[r].append(a2)
    final_ag2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_ex, answer2_ex = await final_ag2(
        [taskInfo, thinking1_class, answer1_class] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2.1: Provide final example scenarios." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True)
    sub_tasks.append(f"Stage2_Subtask1 output: thinking - {thinking2_ex.content}; answer - {answer2_ex.content}")
    subtask_desc2['response'] = {"thinking": thinking2_ex, "answer": answer2_ex}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])
    # Stage 3, Sub-task 1: Prioritize (Debate)
    debate_instr3 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction3 = "Sub-task 3.1: Evaluate and prioritize the error sources based on their typical frequency in genomics data analyses and the subtlety of their detection." + debate_instr3
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R3 = self.max_round
    all_thinking3 = [[] for _ in range(R3)]
    all_answer3 = [[] for _ in range(R3)]
    subtask_desc3_1 = {"subtask_id": "stage3_subtask1", "instruction": debate_instruction3, "context": ["user query", "example thinking", "example answer"], "agent_collaboration": "Debate"}
    for r in range(R3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                t31, a31 = await agent([taskInfo, thinking2_ex, answer2_ex], debate_instruction3, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2_ex, answer2_ex] + all_thinking3[r-1] + all_answer3[r-1]
                t31, a31 = await agent(inputs3, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t31.content}; answer: {a31.content}")
            all_thinking3[r].append(t31)
            all_answer3[r].append(a31)
    final_ag3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_pri, answer3_pri = await final_ag3_1(
        [taskInfo, thinking2_ex, answer2_ex] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3.1: Provide final prioritization." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True)
    sub_tasks.append(f"Stage3_Subtask1 output: thinking - {thinking3_pri.content}; answer - {answer3_pri.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_pri, "answer": answer3_pri}
    logs.append(subtask_desc3_1)
    print("Step 4: ", sub_tasks[-1])
    # Stage 3, Sub-task 2: Match to choices (SC_CoT)
    cot_sc_instruction3_2 = "Sub-task 3.2: Match the top-priority error sources to the provided answer choices and select the correct combination."
    N3_2 = self.max_sc
    sc_agents3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3_2)]
    possible_thinkings3_2 = []
    possible_answers3_2 = []
    subtask_desc3_2 = {"subtask_id": "stage3_subtask2", "instruction": cot_sc_instruction3_2, "context": ["user query", "prioritization thinking", "prioritization answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N3_2):
        t32, a32 = await sc_agents3_2[i]([taskInfo, thinking3_pri, answer3_pri], cot_sc_instruction3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents3_2[i].id}, matching to choices, thinking: {t32.content}; answer: {a32.content}")
        possible_thinkings3_2.append(t32)
        possible_answers3_2.append(a32)
    final_instr3_2 = "Given all the above thinking and answers, find the most consistent and correct selection among the answer choices."
    final_decision_agent3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_match, answer3_match = await final_decision_agent3_2(
        [taskInfo, thinking3_pri, answer3_pri] + possible_thinkings3_2 + possible_answers3_2,
        "Sub-task 3.2: Synthesize and choose the most consistent selection." + final_instr3_2,
        is_sub_task=True)
    sub_tasks.append(f"Stage3_Subtask2 output: thinking - {thinking3_match.content}; answer - {answer3_match.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_match, "answer": answer3_match}
    logs.append(subtask_desc3_2)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3_match, answer3_match, sub_tasks, agents)
    return final_answer, logs