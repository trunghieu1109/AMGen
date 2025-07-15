async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and summarize key information (SC_CoT)
    sc1_instruction = (
        "Sub-task 1: Extract and summarize all key information: identify the target molecule’s substituents and their positions (ethoxy at C1, tert-butyl at C2, nitro at C3), define numbering convention, state starting material, list all nine available reagents from the choices, and record overall constraints.")
    N1 = self.max_sc
    sc1_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_think1 = []
    possible_ans1 = []
    subtask1_desc = {
        "subtask_id": "subtask_1",
        "instruction": sc1_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await sc1_agents[i]([taskInfo], sc1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc1_agents[i].id}, extracting info, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_think1.append(thinking1_i)
        possible_ans1.append(answer1_i)
    dec1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await dec1([taskInfo] + possible_think1 + possible_ans1, "Sub-task 1: Select the most consistent extracted information.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask1_desc['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask1_desc)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Classify reagents (SC_CoT)
    sc2_instruction = (
        "Sub-task 2: Classify each of the nine reagents by reaction type (Friedel–Crafts alkylation, nitration, sulfonation, reduction, diazotization, etherification) and annotate their directing/deactivating or blocking/deblocking roles on the benzene ring.")
    N2 = self.max_sc
    sc2_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_think2 = []
    possible_ans2 = []
    subtask2_desc = {
        "subtask_id": "subtask_2",
        "instruction": sc2_instruction,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2_i, answer2_i = await sc2_agents[i]([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc2_agents[i].id}, classifying reagents, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_think2.append(thinking2_i)
        possible_ans2.append(answer2_i)
    dec2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await dec2([taskInfo, thinking1, answer1] + possible_think2 + possible_ans2, "Sub-task 2: Select the most consistent classification of reagents.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask2_desc['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask2_desc)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Develop regioselectivity strategy (Debate)
    debate3_instruction = (
        "Sub-task 3: Develop a detailed regioselectivity strategy: confirm numbering, analyze directing effects, justify blocking steps, and outline the conceptual step sequence.")
    debate3_instruction += " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate3_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds3 = self.max_round
    all_think3 = [[] for _ in range(rounds3)]
    all_ans3 = [[] for _ in range(rounds3)]
    subtask3_desc = {
        "subtask_id": "subtask_3",
        "instruction": debate3_instruction,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(rounds3):
        for agent in debate3_agents:
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate3_instruction, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_think3[r-1] + all_ans3[r-1]
                thinking3_i, answer3_i = await agent(inputs3, debate3_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_think3[r].append(thinking3_i)
            all_ans3[r].append(answer3_i)
    dec3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final3_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await dec3([taskInfo, thinking1, answer1, thinking2, answer2] + all_think3[-1] + all_ans3[-1], "Sub-task 3: Develop regioselectivity strategy. " + final3_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask3_desc['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask3_desc)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4.1: Structured mapping (SC_CoT)
    sc4_instruction = (
        "Sub-task 4.1: For each of the four answer choices, list the nine numbered reagents in order and map each to the conceptual steps (blocking, alkylation, nitration‐step-1, etc.), ensuring all conceptual steps appear.")
    N4 = self.max_sc
    sc4_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    poss_think4 = []
    poss_ans4 = []
    subtask4_1_desc = {
        "subtask_id": "subtask_4_1",
        "instruction": sc4_instruction,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4_i, answer4_i = await sc4_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], sc4_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc4_agents[i].id}, mapping sequences, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        poss_think4.append(thinking4_i)
        poss_ans4.append(answer4_i)
    dec4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await dec4_1([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + poss_think4 + poss_ans4, "Sub-task 4.1: Select the most consistent structured mapping.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask4_1_desc['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask4_1_desc)
    print("Step 4.1: ", sub_tasks[-1])

    # Stage 4.2: Validation via cross-check (Reflexion)
    reflect4_2 = (
        "Sub-task 4.2: Validate the mapped sequences via cross-check: challenge and confirm that each mapping leads to the correct regiochemical outcome, flag any misorders, and ensure the final pattern matches the target molecule.")
    reflect4_2 += " Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs4_2 = [taskInfo, thinking4_1, answer4_1]
    subtask4_2_desc = {
        "subtask_id": "subtask_4_2",
        "instruction": reflect4_2,
        "context": ["user query", thinking4_1, answer4_1],
        "agent_collaboration": "Reflexion"
    }
    thinking4_2, answer4_2 = await cot4_2(inputs4_2, reflect4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot4_2.id}, initial validation thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic4_2([taskInfo, thinking4_2, answer4_2], (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        ), i, is_sub_task=True)
        agents.append(f"Critic agent {critic4_2.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        thinking4_2, answer4_2 = await cot4_2([taskInfo, thinking4_2, answer4_2, feedback4], reflect4_2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4_2.id}, refinement thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask4_2_desc['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask4_2_desc)
    print("Step 4.2: ", sub_tasks[-1])

    # Stage 5: Evaluate and select optimal route (Debate)
    debate5_instruction = (
        "Sub-task 5: Evaluate the fully validated sequences against synthetic criteria (regiocontrol, yield potential, step-count) and select the optimal nine-step route, providing clear justification.")
    debate5_instruction += " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate5_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds5 = self.max_round
    all_think5 = [[] for _ in range(rounds5)]
    all_ans5 = [[] for _ in range(rounds5)]
    subtask5_desc = {
        "subtask_id": "subtask_5",
        "instruction": debate5_instruction,
        "context": ["user query", thinking4_2, answer4_2],
        "agent_collaboration": "Debate"
    }
    for r in range(rounds5):
        for agent in debate5_agents:
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4_2, answer4_2], debate5_instruction, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4_2, answer4_2] + all_think5[r-1] + all_ans5[r-1]
                thinking5_i, answer5_i = await agent(inputs5, debate5_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_think5[r].append(thinking5_i)
            all_ans5[r].append(answer5_i)
    dec5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final5_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await dec5([taskInfo, thinking4_2, answer4_2] + all_think5[-1] + all_ans5[-1], "Sub-task 5: Evaluate and select optimal route. " + final5_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask5_desc['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask5_desc)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs