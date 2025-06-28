async def forward_76(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_debate_instruction_1a = (
        "Sub-task 1a: Identify and analyze the most acidic C–H bonds and reactive sites in (((3-methylbut-2-en-1-yl)oxy)methyl)benzene under BuLi conditions. "
        "Consider organolithium reactivity, pKa values, and possible deprotonation or cleavage pathways. "
        "Provide detailed mechanistic reasoning and cite relevant chemical principles."
    )
    debate_agents_1a = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1a = self.max_round
    all_thinking_1a = [[] for _ in range(N_max_1a)]
    all_answer_1a = [[] for _ in range(N_max_1a)]
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_debate_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1a):
        for i, agent in enumerate(debate_agents_1a):
            if r == 0:
                thinking_1a, answer_1a = await agent([taskInfo], cot_debate_instruction_1a, r, is_sub_task=True)
            else:
                input_infos_1a = [taskInfo] + all_thinking_1a[r-1] + all_answer_1a[r-1]
                thinking_1a, answer_1a = await agent(input_infos_1a, cot_debate_instruction_1a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing acidic C-H bonds and reactive sites, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
            all_thinking_1a[r].append(thinking_1a)
            all_answer_1a[r].append(answer_1a)
    final_decision_agent_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await final_decision_agent_1a([taskInfo] + all_thinking_1a[-1] + all_answer_1a[-1], "Sub-task 1a: Make final decision on the most acidic C-H bonds and reactive sites under BuLi conditions.", is_sub_task=True)
    agents.append(f"Final Decision agent on subtask_1a, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a["response"] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_sc_instruction_1b = (
        "Sub-task 1b: Predict all plausible reaction pathways after BuLi treatment of (((3-methylbut-2-en-1-yl)oxy)methyl)benzene, "
        "including deprotonation, benzyl ether cleavage, and possible rearrangements. Provide mechanistic justification for each pathway. "
        "Use self-consistency chain-of-thought to generate multiple plausible pathways and select the most chemically sound."
    )
    N_sc_1b = self.max_sc
    cot_agents_1b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_1b)
    ]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1b):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo, thinking_1a, answer_1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, predicting plausible pathways after BuLi treatment, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b.content)
        thinkingmapping_1b[answer_1b.content] = thinking_1b
        answermapping_1b[answer_1b.content] = answer_1b
    answer_1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinkingmapping_1b[answer_1b_content]
    answer_1b = answermapping_1b[answer_1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b["response"] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])

    cot_sc_instruction_1c = (
        "Sub-task 1c: Assign the stereochemistry and regiochemistry of the major product(s) formed after protonation (H+) following BuLi treatment, "
        "based on the most plausible pathway(s) identified in subtask 1b. Provide detailed mechanistic reasoning and stereochemical justification. "
        "Use self-consistency chain-of-thought to generate multiple stereochemical assignments and select the most consistent."
    )
    N_sc_1c = self.max_sc
    cot_agents_1c = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_1c)
    ]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1c):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, assigning stereochemistry and regiochemistry of major product(s), thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    answer_1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[answer_1c_content]
    answer_1c = answermapping_1c[answer_1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c["response"] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])

    cot_instruction_2a = (
        "Sub-task 2a: Perform detailed structural analysis of 3,4,5,7,8,9-hexamethyl-1,11-dimethylene-2,6,10,11,11a,11b-hexahydro-1H-benzo[cd]indeno[7,1-gh]azulene, "
        "including numbering, identification of 1,5-diene moieties, and relevant stereochemical features. Provide detailed mechanistic context."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, performing structural analysis of substrate for Cope rearrangement, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a["response"] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_sc_instruction_2b = (
        "Sub-task 2b: Map the Cope rearrangement stepwise on the substrate analyzed in subtask 2a, "
        "including transition states or intermediates, to predict structural changes, focusing on double bond migration and ring system rearrangement. "
        "Use self-consistency chain-of-thought to generate multiple mechanistic mappings and select the most consistent."
    )
    N_sc_2b = self.max_sc
    cot_agents_2b = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_2b)
    ]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2b):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, mapping Cope rearrangement stepwise, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b["response"] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_sc_instruction_2c = (
        "Sub-task 2c: Determine the stereochemical outcome and hydrogenation state (e.g., tetrahydro vs. hexahydro) of the product after heat-induced Cope rearrangement, "
        "based on the mechanistic mapping in subtask 2b. Provide mechanistic and energetic justification. "
        "Use self-consistency chain-of-thought to generate multiple stereochemical assignments and select the most consistent."
    )
    N_sc_2c = self.max_sc
    cot_agents_2c = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_2c)
    ]
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2c):
        thinking_2c, answer_2c = await cot_agents_2c[i]([taskInfo, thinking_2b, answer_2b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, determining stereochemical outcome and hydrogenation state, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
        possible_answers_2c.append(answer_2c.content)
        thinkingmapping_2c[answer_2c.content] = thinking_2c
        answermapping_2c[answer_2c.content] = answer_2c
    answer_2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking_2c = thinkingmapping_2c[answer_2c_content]
    answer_2c = answermapping_2c[answer_2c_content]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c["response"] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Generate multiple plausible product predictions for both reactions (A and B) using Self-Consistency Chain-of-Thought (SC CoT) "
        "to cross-validate mechanistic pathways and stereochemical assignments from subtasks 1c and 2c. "
        "Synthesize the most consistent and chemically sound predictions."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_3)
    ]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_1c, answer_1c, thinking_2c, answer_2c], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, generating plausible product predictions for reactions A and B, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3.content)
        thinkingmapping_3[answer_3.content] = thinking_3
        answermapping_3[answer_3.content] = answer_3
    answer_3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinkingmapping_3[answer_3_content]
    answer_3 = answermapping_3[answer_3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3["response"] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_reflect_instruction_4 = (
        "Sub-task 4: Critically evaluate and compare the predicted products from subtask 3 against the four given multiple-choice options. "
        "Incorporate confidence levels, uncertainties, and mechanistic consistency to select the most chemically consistent answer. "
        "Use Reflexion with iterative critic feedback to refine the selection."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking_3, answer_3]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating predicted products against choices, thinking: {thinking_4.content}; answer: {answer_4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking_4, answer_4], "Critically evaluate the product comparison and selection, providing limitations and confidence.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking_4, answer_4, feedback])
        thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining product comparison and selection, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4["response"] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
