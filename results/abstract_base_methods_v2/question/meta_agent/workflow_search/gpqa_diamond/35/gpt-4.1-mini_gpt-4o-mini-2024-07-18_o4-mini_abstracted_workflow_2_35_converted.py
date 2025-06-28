async def forward_35(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = (
        "Sub-task 1a: Convert the given absorption line energy of 3.9 micro electron volts (3.9 × 10^-6 eV) "
        "into its corresponding frequency (Hz) and wavelength (meters) using fundamental physics formulas (E = hf and λ = c/f)."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, converting energy to frequency and wavelength, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Compile a concise reference table of standard interstellar medium (ISM) absorption lines, "
        "including their characteristic energies, frequencies, and associated ISM phases (e.g., 21 cm HI hyperfine line at ~5.9 μeV for cold atomic ISM, molecular rotational/vibrational transitions for cold molecular ISM, etc.)."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, compiling ISM absorption line catalog, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_sc_instruction_1c = (
        "Sub-task 1c: Compare the calculated frequency and wavelength from Sub-task 1a with the reference ISM absorption lines from Sub-task 1b "
        "to identify candidate ISM phases that could produce an absorption line near 3.9 μeV, explicitly including the 21 cm HI line and other relevant transitions."
    )
    N_sc_1c = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1c)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1c):
        thinking_1c, answer_1c = await cot_agents_1c[i]([taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, identifying candidate ISM phases, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    answer_1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[answer_1c_content]
    answer_1c = answermapping_1c[answer_1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_reflect_instruction_2a = (
        "Sub-task 2a: Critically evaluate and rank the candidate ISM phases identified in Sub-task 1c by considering astrophysical plausibility, "
        "typical ISM conditions in the Milky Way, and proximity of the observed energy to known standard lines, applying reflexion and self-consistency checks to avoid confirmation bias."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2a = self.max_round
    cot_inputs_2a = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_1c, answer_1c]
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_reflect_instruction_2a,
        "context": ["user query", "thinking and answer of subtasks 1a, 1b, 1c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2a, answer_2a = await cot_agent_2a(cot_inputs_2a, cot_reflect_instruction_2a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, evaluating and ranking candidate ISM phases, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    for i in range(N_max_2a):
        feedback, correct = await critic_agent_2a([taskInfo, thinking_2a, answer_2a],
                                                 "Critically evaluate the ranking and astrophysical plausibility of candidate ISM phases and provide limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2a.extend([thinking_2a, answer_2a, feedback])
        thinking_2a, answer_2a = await cot_agent_2a(cot_inputs_2a, cot_reflect_instruction_2a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, refining evaluation and ranking, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    debate_instruction_2b = (
        "Sub-task 2b: Conduct a structured debate comparing the top candidate ISM phases (e.g., cold atomic vs. cold molecular ISM) "
        "as the source of the observed absorption line, with a critic agent adjudicating to select the most likely ISM phase based on quantitative and qualitative evidence."
    )
    debate_agents_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2b = self.max_round
    all_thinking_2b = [[] for _ in range(N_max_2b)]
    all_answer_2b = [[] for _ in range(N_max_2b)]
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": debate_instruction_2b,
        "context": ["user query", "thinking and answer of subtask 2a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2b):
        for i, agent in enumerate(debate_agents_2b):
            if r == 0:
                thinking_2b, answer_2b = await agent([taskInfo, thinking_2a, answer_2a], debate_instruction_2b, r, is_sub_task=True)
            else:
                input_infos_2b = [taskInfo, thinking_2a, answer_2a] + all_thinking_2b[r-1] + all_answer_2b[r-1]
                thinking_2b, answer_2b = await agent(input_infos_2b, debate_instruction_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating ISM phase source, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
            all_thinking_2b[r].append(thinking_2b)
            all_answer_2b[r].append(answer_2b)
    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2b, answer_2b = await final_decision_agent_2b([taskInfo] + all_thinking_2b[-1] + all_answer_2b[-1],
                                                          "Sub-task 2b: Make final decision on the most likely ISM phase based on debate.",
                                                          is_sub_task=True)
    agents.append(f"Final Decision agent, adjudicating debate, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_2c = (
        "Sub-task 2c: Map the adjudicated ISM phase from Sub-task 2b to the corresponding multiple-choice option "
        "(A: Cold molecular ISM, B: Cold atomic ISM, C: Warm atomic ISM, D: Warm molecular ISM) and prepare the final answer in the required output format."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking and answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    thinking_2c, answer_2c = await cot_agent_2c([taskInfo, thinking_2b, answer_2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, mapping ISM phase to multiple-choice and finalizing answer, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2c, answer_2c, sub_tasks, agents)
    return final_answer, logs
