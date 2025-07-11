async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Parse the query to identify that three separate Michael addition reactions (A, B, and C) must be analysed for their reactants and major products."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: For reaction A (dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate with NaOEt/EtOH), identify the nucleophile and the electrophilic Michael acceptor."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "response of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying nucleophile and acceptor for reaction A, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
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

    cot_sc_instruction = "Sub-task 3: Predict the structure and IUPAC-style name of the major Michael addition product for reaction A, considering resonance and final ester configuration."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_2", "answer of subtask_2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, predicting product structure and name for reaction A, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_answers.append(answer3_i.content)
        thinkingmapping[answer3_i.content] = thinking3_i
        answermapping[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3_content]
    answer3 = answermapping[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction = "Sub-task 4: For reaction B (1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile with MeOH, H3O+), identify the enamine nucleophile and the Michael acceptor."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying nucleophile and acceptor for reaction B, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 5: Predict the major final product of reaction B after hydrolysis, including its keto and nitrile functionalities, and give its systematic name."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5_i, answer5_i = await cot_agents[i]([taskInfo, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, predicting reaction B product, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers.append(answer5_i.content)
        thinkingmapping[answer5_i.content] = thinking5_i
        answermapping[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5_content]
    answer5 = answermapping[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction = "Sub-task 6: For reaction C (unknown C + but-3-en-2-one with KOH/H2O), recognize that cyclohexane-1,3-dione (or its enolate) is the nucleophile and but-3-en-2-one is the acceptor, and identify the missing reactant C."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying missing reactant C, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 7: Predict the major final product of reaction C (2-(3-oxobutyl)cyclohexane-1,3-dione) and confirm the identity of reactant C as cyclohexane-1,3-dione."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking7_i, answer7_i = await cot_agents[i]([taskInfo, thinking6, answer6], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, predicting reaction C product and confirming C, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible_answers.append(answer7_i.content)
        thinkingmapping[answer7_i.content] = thinking7_i
        answermapping[answer7_i.content] = answer7_i
    answer7_content = Counter(possible_answers).most_common(1)[0][0]
    thinking7 = thinkingmapping[answer7_content]
    answer7 = answermapping[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction = "Sub-task 8: Compile the predicted product names for reactions A, B, and C, along with the identity of reactant C, into a set of three named compounds."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction, "context": ["user query","thinking of subtask_3","answer of subtask_3","thinking of subtask_5","answer of subtask_5","thinking of subtask_7","answer of subtask_7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent([taskInfo, thinking3, answer3, thinking5, answer5, thinking7, answer7], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, compiling product names and reactant C identity, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    debate_instruction = "Sub-task 9: Compare the compiled predictions against the four given multiple-choice answer sets and select the matching choice letter."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking9 = [[] for _ in range(self.max_round)]
    all_answer9 = [[] for _ in range(self.max_round)]
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": debate_instruction, "context": ["user query","thinking of subtask_8","answer of subtask_8"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking9_i, answer9_i = await agent([taskInfo, thinking8, answer8], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9_i, answer9_i = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting choice letter, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
            all_thinking9[r].append(thinking9_i)
            all_answer9[r].append(answer9_i)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the matching choice letter from multiple-choice options.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent.id}, making final choice decision, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9["response"] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs