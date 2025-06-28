async def forward_139(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    debate_roles = self.debate_role if hasattr(self, 'debate_role') else ["Pro", "Con"]

    subtask_1a_instruction = (
        "Sub-task 1a: Identify all plausible candidate gases for gas W based on the clue that its molecule contains the same number of neutrons and protons, "
        "considering isotopic possibilities such as N2, H2, and D2, and their nuclear compositions."
    )
    subtask_1a_desc = {
        "subtask_id": "subtask_1a",
        "instruction": subtask_1a_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1a = []
    thinking_map_1a = {}
    answer_map_1a = {}
    for i in range(self.max_sc):
        thinking1a, answer1a = await cot_sc_agents_1a[i]([taskInfo], subtask_1a_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1a[i].id}, identifying candidate gases for gas W, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinking_map_1a[answer1a.content] = thinking1a
        answer_map_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinking_map_1a[answer1a_content]
    answer1a = answer_map_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_1a_desc["response"] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_1a_desc)
    print("Step 1a: ", sub_tasks[-1])

    subtask_1b_instruction = (
        "Sub-task 1b: Identify all plausible candidate liquids for liquid Y, focusing on isotopic variants of water such as H2O and D2O, "
        "and other likely liquids, using melting points and isotopic clues provided in the query."
    )
    subtask_1b_desc = {
        "subtask_id": "subtask_1b",
        "instruction": subtask_1b_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1b = []
    thinking_map_1b = {}
    answer_map_1b = {}
    for i in range(self.max_sc):
        thinking1b, answer1b = await cot_sc_agents_1b[i]([taskInfo], subtask_1b_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1b[i].id}, identifying candidate liquids for liquid Y, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinking_map_1b[answer1b.content] = thinking1b
        answer_map_1b[answer1b.content] = answer1b
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinking_map_1b[answer1b_content]
    answer1b = answer_map_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_1b_desc["response"] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_1b_desc)
    print("Step 1b: ", sub_tasks[-1])

    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_1c_instruction = (
        "Sub-task 1c: Match candidate gases from Sub-task 1a and liquids from Sub-task 1b against the physical data provided, "
        "including the melting point of substance B near 277 K and isotopic composition clues, to narrow down the identities of gas W and liquid Y."
    )
    subtask_1c_desc = {
        "subtask_id": "subtask_1c",
        "instruction": subtask_1c_instruction,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1a, answer1a, thinking1b, answer1b], subtask_1c_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, matching candidates for gas W and liquid Y, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_1c_desc["response"] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_1c_desc)
    print("Step 1c: ", sub_tasks[-1])

    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_1d_instruction = (
        "Sub-task 1d: Using the narrowed candidates for gas W and liquid Y from Sub-task 1c, identify Substance X by analyzing its isotopic composition, "
        "reaction context, and known reagents in organic chemistry that incorporate heavier isotopes."
    )
    subtask_1d_desc = {
        "subtask_id": "subtask_1d",
        "instruction": subtask_1d_instruction,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "CoT"
    }
    thinking1d, answer1d = await cot_agent_1d([taskInfo, thinking1c, answer1c], subtask_1d_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1d.id}, identifying Substance X, thinking: {thinking1d.content}; answer: {answer1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking1d.content}; answer - {answer1d.content}")
    subtask_1d_desc["response"] = {"thinking": thinking1d, "answer": answer1d}
    logs.append(subtask_1d_desc)
    print("Step 1d: ", sub_tasks[-1])

    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_2a_instruction = (
        "Sub-task 2a: Identify precipitate G formed in the reaction of Substance X with liquid Y, using the reaction context and the property that heating G releases substance B with a melting point near 277 K."
    )
    subtask_2a_desc = {
        "subtask_id": "subtask_2a",
        "instruction": subtask_2a_instruction,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "CoT"
    }
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1d, answer1d], subtask_2a_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, identifying precipitate G, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_2a_desc["response"] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_2a_desc)
    print("Step 2a: ", sub_tasks[-1])

    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_2b_instruction = (
        "Sub-task 2b: Confirm the identity of substance B released upon heating precipitate G by cross-referencing its melting point and isotopic composition."
    )
    subtask_2b_desc = {
        "subtask_id": "subtask_2b",
        "instruction": subtask_2b_instruction,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking2a, answer2a], subtask_2b_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, confirming substance B, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_2b_desc["response"] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_2b_desc)
    print("Step 2b: ", sub_tasks[-1])

    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_2c_instruction = (
        "Sub-task 2c: Analyze the product formed from the reaction of Substance X with a certain keto acid, focusing on the product containing 2 oxygen atoms, "
        "to validate or refine the identification of Substance X and its close analog used as a reagent in organic chemistry."
    )
    subtask_2c_desc = {
        "subtask_id": "subtask_2c",
        "instruction": subtask_2c_instruction,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "CoT"
    }
    thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking1d, answer1d], subtask_2c_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, analyzing keto acid reaction product, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_2c_desc["response"] = {"thinking": thinking2c, "answer": answer2c}
    logs.append(subtask_2c_desc)
    print("Step 2c: ", sub_tasks[-1])

    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_3a = self.max_round
    all_thinking_3a = [[] for _ in range(N_max_3a)]
    all_answer_3a = [[] for _ in range(N_max_3a)]
    subtask_3a_instruction = (
        "Sub-task 3a: List all elements present in Substance X and its very close analog, including all relevant isotopes (heavier and lighter), "
        "based on the confirmed identification from previous subtasks."
    )
    subtask_3a_desc = {
        "subtask_id": "subtask_3a",
        "instruction": subtask_3a_instruction,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking1d, answer1d, thinking2c, answer2c], subtask_3a_instruction, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking1d, answer1d, thinking2c, answer2c] + all_thinking_3a[r-1] + all_answer_3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, subtask_3a_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, listing elements and isotopes, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking_3a[r].append(thinking3a)
            all_answer_3a[r].append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + all_thinking_3a[-1] + all_answer_3a[-1], "Sub-task 3a: Make final decision on elements and isotopes list.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing elements and isotopes list, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_3a_desc["response"] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_3a_desc)
    print("Step 3a: ", sub_tasks[-1])

    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_3b_instruction = "Sub-task 3b: Identify the lightest and heaviest elements present in Substance X from the comprehensive list generated in Sub-task 3a."
    subtask_3b_desc = {
        "subtask_id": "subtask_3b",
        "instruction": subtask_3b_instruction,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], subtask_3b_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, identifying lightest and heaviest elements, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_3b_desc["response"] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_3b_desc)
    print("Step 3b: ", sub_tasks[-1])

    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_3c_instruction = (
        "Sub-task 3c: Calculate the cumulative atomic masses of all heavier and lighter isotopes of the lightest and heaviest elements present in Substance X, "
        "summing the isotopic masses for each element."
    )
    subtask_3c_desc = {
        "subtask_id": "subtask_3c",
        "instruction": subtask_3c_instruction,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking3b, answer3b], subtask_3c_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, calculating cumulative atomic masses, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_3c_desc["response"] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_3c_desc)
    print("Step 3c: ", sub_tasks[-1])

    cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_3d_instruction = "Sub-task 3d: Sum the cumulative atomic masses of the lightest and heaviest elements to obtain the final value required by the query."
    subtask_3d_desc = {
        "subtask_id": "subtask_3d",
        "instruction": subtask_3d_instruction,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "CoT"
    }
    thinking3d, answer3d = await cot_agent_3d([taskInfo, thinking3c, answer3c], subtask_3d_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3d.id}, summing cumulative atomic masses, thinking: {thinking3d.content}; answer: {answer3d.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_3d_desc["response"] = {"thinking": thinking3d, "answer": answer3d}
    logs.append(subtask_3d_desc)
    print("Step 3d: ", sub_tasks[-1])

    cot_agent_3e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_3e_instruction = (
        "Sub-task 3e: Compare the calculated sum with the provided multiple-choice options and select the correct answer, "
        "returning only the letter choice (A, B, C, or D) as per the output format requirement."
    )
    subtask_3e_desc = {
        "subtask_id": "subtask_3e",
        "instruction": subtask_3e_instruction,
        "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d", "choices: A=29, B=25, C=35, D=31"],
        "agent_collaboration": "CoT"
    }
    thinking3e, answer3e = await cot_agent_3e([taskInfo, thinking3d, answer3d], subtask_3e_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3e.id}, selecting correct answer, thinking: {thinking3e.content}; answer: {answer3e.content}")
    sub_tasks.append(f"Sub-task 3e output: thinking - {thinking3e.content}; answer - {answer3e.content}")
    subtask_3e_desc["response"] = {"thinking": thinking3e, "answer": answer3e}
    logs.append(subtask_3e_desc)
    print("Step 3e: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3e, answer3e, sub_tasks, agents)
    return final_answer, logs
