async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_agents_num = self.max_sc
    max_reflexion_rounds = self.max_round
    debate_rounds = self.max_round
    debate_roles = self.debate_role
    # Stage 1: Enumerate neutrino sources, map spectra to bands, collect fluxes, and self-consistency check
    # Subtask 1a: Enumerate all solar neutrino production branches with spectra and endpoints
    instruction_1a = (
        "Sub-task 1a: Enumerate all solar neutrino production branches (pp-I, pp-II, pp-III, pep, 7Be, 8B, "
        "and CNO cycle branches such as 13N, 15O, 17F), including their characteristic neutrino energy spectra and spectral endpoints, "
        "to establish a comprehensive baseline of neutrino sources relevant to the energy bands 700-800 keV and 800-900 keV."
    )
    subtask_1a_desc = {
        "subtask_id": "subtask_1a",
        "instruction": instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, enumerating neutrino branches and spectra, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_1a_desc['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_1a_desc)
    print("Step 1a: ", sub_tasks[-1])
    # Subtask 1b: Map each branch's energy spectrum onto the specified energy bands
    instruction_1b = (
        "Sub-task 1b: Map each neutrino production branch's energy spectrum onto the specified energy bands (700-800 keV and 800-900 keV), "
        "identifying which branches contribute neutrinos to each band and estimating the approximate fraction or shape of their flux within these bands."
    )
    subtask_1b_desc = {
        "subtask_id": "subtask_1b",
        "instruction": instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_1b = []
    thinking_map_1b = {}
    answer_map_1b = {}
    for i in range(cot_sc_agents_num):
        thinking_i, answer_i = await cot_agents_1b[i]([taskInfo, thinking_1a, answer_1a], instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, mapping spectra to energy bands, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1b.append(answer_i.content)
        thinking_map_1b[answer_i.content] = thinking_i
        answer_map_1b[answer_i.content] = answer_i
    answer_1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinking_map_1b[answer_1b_content]
    answer_1b = answer_map_1b[answer_1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_1b_desc['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_1b_desc)
    print("Step 1b: ", sub_tasks[-1])
    # Subtask 1c: Collect quantitative reference flux values for each neutrino source
    instruction_1c = (
        "Sub-task 1c: Collect and compile quantitative reference flux values (in neutrinos per cm^2 per second) for each neutrino source branch under normal solar conditions, "
        "including pp-chain and CNO cycle contributions, to enable numerical flux estimations in the energy bands."
    )
    subtask_1c_desc = {
        "subtask_id": "subtask_1c",
        "instruction": instruction_1c,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1a, answer_1a], instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, collecting quantitative flux data, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_1c_desc['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_1c_desc)
    print("Step 1c: ", sub_tasks[-1])
    # Subtask 1d: Self-consistency check of source list, spectral mappings, and flux values
    instruction_1d = (
        "Sub-task 1d: Perform a self-consistency check by cross-validating the completeness and accuracy of the neutrino source list, their spectral mappings, and flux values, "
        "ensuring no relevant sources or spectral overlaps are omitted before proceeding."
    )
    subtask_1d_desc = {
        "subtask_id": "subtask_1d",
        "instruction": instruction_1d,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_1d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs_1d = [taskInfo, thinking_1b, answer_1b, thinking_1c, answer_1c]
    thinking_1d, answer_1d = await cot_agent_1d(inputs_1d, instruction_1d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1d.id}, self-consistency check of neutrino sources and fluxes, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    for i in range(max_reflexion_rounds):
        feedback_1d, correct_1d = await critic_agent_1d([taskInfo, thinking_1d, answer_1d],
                                                      "please review the self-consistency check and provide its limitations.",
                                                      i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1d.id}, providing feedback, thinking: {feedback_1d.content}; answer: {correct_1d.content}")
        if correct_1d.content == "True":
            break
        inputs_1d.extend([thinking_1d, answer_1d, feedback_1d])
        thinking_1d, answer_1d = await cot_agent_1d(inputs_1d, instruction_1d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1d.id}, refining self-consistency check, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_1d_desc['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_1d_desc)
    print("Step 1d: ", sub_tasks[-1])
    # Stage 2: Analyze pp-III branch, effect of stopping, calculate adjusted fluxes, compute ratio, and reflect
    # Subtask 2a: Analyze pp-III branch energy range and flux contribution
    instruction_2a = (
        "Sub-task 2a: Analyze the specific neutrino energy range and flux contribution of the pp-III branch, clarifying its role in the 700-900 keV bands and distinguishing it from other branches, "
        "to understand the impact of its hypothetical cessation."
    )
    subtask_2a_desc = {
        "subtask_id": "subtask_2a",
        "instruction": instruction_2a,
        "context": ["user query", "thinking of subtask_1d", "answer of subtask_1d"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_agents_num)]
    possible_answers_2a = []
    thinking_map_2a = {}
    answer_map_2a = {}
    for i in range(cot_sc_agents_num):
        thinking_i, answer_i = await cot_agents_2a[i]([taskInfo, thinking_1d, answer_1d], instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, analyzing pp-III branch energy and flux, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2a.append(answer_i.content)
        thinking_map_2a[answer_i.content] = thinking_i
        answer_map_2a[answer_i.content] = answer_i
    answer_2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinking_map_2a[answer_2a_content]
    answer_2a = answer_map_2a[answer_2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_2a_desc['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_2a_desc)
    print("Step 2a: ", sub_tasks[-1])
    # Subtask 2b: Determine effect on neutrino flux 8.5 minutes after pp-III stops
    instruction_2b = (
        "Sub-task 2b: Determine the effect on the neutrino flux reaching Earth 8.5 minutes after the sudden stop of the pp-III branch in the solar core, "
        "considering neutrino travel time and assuming all other branches remain unchanged."
    )
    subtask_2b_desc = {
        "subtask_id": "subtask_2b",
        "instruction": instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs_2b = [taskInfo, thinking_2a, answer_2a]
    thinking_2b, answer_2b = await cot_agent_2b(inputs_2b, instruction_2b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, determining flux effect after pp-III stops, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    for i in range(max_reflexion_rounds):
        feedback_2b, correct_2b = await critic_agent_2b([taskInfo, thinking_2b, answer_2b],
                                                      "please review the analysis of the flux effect after stopping pp-III and provide its limitations.",
                                                      i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2b.id}, providing feedback, thinking: {feedback_2b.content}; answer: {correct_2b.content}")
        if correct_2b.content == "True":
            break
        inputs_2b.extend([thinking_2b, answer_2b, feedback_2b])
        thinking_2b, answer_2b = await cot_agent_2b(inputs_2b, instruction_2b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, refining flux effect analysis, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_2b_desc['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_2b_desc)
    print("Step 2b: ", sub_tasks[-1])
    # Subtask 2c: Calculate adjusted neutrino fluxes in the two energy bands after pp-III stops
    instruction_2c = (
        "Sub-task 2c: Calculate the adjusted neutrino fluxes in the two energy bands (700-800 keV and 800-900 keV) after the pp-III branch stops, "
        "using the quantitative flux data and spectral mappings, and accounting for the removal of pp-III contributions."
    )
    subtask_2c_desc = {
        "subtask_id": "subtask_2c",
        "instruction": instruction_2c,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b", "thinking of subtask_1b", "answer of subtask_1b", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "CoT"
    }
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2c, answer_2c = await cot_agent_2c([taskInfo, thinking_2b, answer_2b, thinking_1b, answer_1b, thinking_1c, answer_1c], instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, calculating adjusted fluxes after pp-III stops, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_2c_desc['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_2c_desc)
    print("Step 2c: ", sub_tasks[-1])
    # Subtask 2d: Compute approximate ratio of flux (band 1) to flux (band 2) after pp-III stops and compare to options
    instruction_2d = (
        "Sub-task 2d: Compute the approximate ratio of flux (band 1: 700-800 keV) to flux (band 2: 800-900 keV) after the pp-III branch stops, "
        "and compare this ratio to the provided multiple-choice options."
    )
    subtask_2d_desc = {
        "subtask_id": "subtask_2d",
        "instruction": instruction_2d,
        "context": ["user query", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "Debate"
    }
    debate_agents_2d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    all_thinking_2d = [[] for _ in range(debate_rounds)]
    all_answer_2d = [[] for _ in range(debate_rounds)]
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_2d):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2c, answer_2c], instruction_2d, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2c, answer_2c] + all_thinking_2d[r-1] + all_answer_2d[r-1]
                thinking_i, answer_i = await agent(input_infos, instruction_2d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing flux ratio and comparing options, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2d[r].append(thinking_i)
            all_answer_2d[r].append(answer_i)
    final_decision_agent_2d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2d, answer_2d = await final_decision_agent_2d([taskInfo] + all_thinking_2d[-1] + all_answer_2d[-1],
                                                          "Sub-task 2d: Make final decision on the approximate flux ratio between band 1 and band 2 after pp-III stops.",
                                                          is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final flux ratio, thinking: {thinking_2d.content}; answer: {answer_2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking_2d.content}; answer - {answer_2d.content}")
    subtask_2d_desc['response'] = {"thinking": thinking_2d, "answer": answer_2d}
    logs.append(subtask_2d_desc)
    print("Step 2d: ", sub_tasks[-1])
    # Subtask 2e: Reflection and verification of calculated flux ratio and assumptions
    instruction_2e = (
        "Sub-task 2e: Conduct a reflection and verification step to review the calculated flux ratio and underlying assumptions, "
        "ensuring no relevant neutrino sources or spectral contributions were overlooked, and confirm the final answer's consistency with the physics of solar neutrino production."
    )
    subtask_2e_desc = {
        "subtask_id": "subtask_2e",
        "instruction": instruction_2e,
        "context": ["user query", "thinking of subtask_2d", "answer of subtask_2d"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_2e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2e = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs_2e = [taskInfo, thinking_2d, answer_2d]
    thinking_2e, answer_2e = await cot_agent_2e(inputs_2e, instruction_2e, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2e.id}, reflecting on flux ratio and assumptions, thinking: {thinking_2e.content}; answer: {answer_2e.content}")
    for i in range(max_reflexion_rounds):
        feedback_2e, correct_2e = await critic_agent_2e([taskInfo, thinking_2e, answer_2e],
                                                      "please review the reflection on flux ratio and confirm correctness.",
                                                      i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2e.id}, providing feedback, thinking: {feedback_2e.content}; answer: {correct_2e.content}")
        if correct_2e.content == "True":
            break
        inputs_2e.extend([thinking_2e, answer_2e, feedback_2e])
        thinking_2e, answer_2e = await cot_agent_2e(inputs_2e, instruction_2e, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2e.id}, refining reflection, thinking: {thinking_2e.content}; answer: {answer_2e.content}")
    sub_tasks.append(f"Sub-task 2e output: thinking - {thinking_2e.content}; answer - {answer_2e.content}")
    subtask_2e_desc['response'] = {"thinking": thinking_2e, "answer": answer_2e}
    logs.append(subtask_2e_desc)
    print("Step 2e: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking_2e, answer_2e, sub_tasks, agents)
    return final_answer, logs
